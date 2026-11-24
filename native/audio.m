// Overtone native audio & DSP bridge for macOS:
//   - Live microphone capture via AVAudioEngine (inputNode tap -> thread-safe mono queue)
//   - Audio file decoding (WAV, MP3, M4A, AAC, FLAC, AIFF, CAF) via AVAudioFile -> mono f32
//   - Playback of mono f32 buffers with range selection, looping, seeking, and volume
//   - 16-bit PCM WAV export
//   - Hardware-accelerated vDSP FFT magnitudes & autocorrelation for spectrogram & Boersma pitch
#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>
#import <AudioToolbox/AudioToolbox.h>
#import <Cocoa/Cocoa.h>
#import <CoreAudio/CoreAudio.h>
#import <math.h>
#import <os/lock.h>
#import <stdatomic.h>
#import <stdint.h>
#import <stdlib.h>
#import <string.h>

// ============================================================================
// CoreAudio Input / Output Device Enumeration & Selection
// ============================================================================

static os_unfair_lock gDevLock = OS_UNFAIR_LOCK_INIT;
static NSMutableArray<NSDictionary *> *gInputDevices = nil;
static NSMutableArray<NSDictionary *> *gOutputDevices = nil;
static AudioDeviceID gSelectedInputDeviceId = 0;   // 0 = System Default
static AudioDeviceID gSelectedOutputDeviceId = 0;  // 0 = System Default
static BOOL gPlayDeviceDirty = NO;

static uint32_t caDeviceChannels(AudioDeviceID devId, AudioObjectPropertyScope scope) {
  AudioObjectPropertyAddress addr = {
    kAudioDevicePropertyStreamConfiguration,
    scope,
    kAudioObjectPropertyElementMain
  };
  UInt32 dataSize = 0;
  if (AudioObjectGetPropertyDataSize(devId, &addr, 0, NULL, &dataSize) != noErr || dataSize == 0) {
    return 0;
  }
  AudioBufferList *bufList = (AudioBufferList *)malloc(dataSize);
  if (!bufList) return 0;
  uint32_t total = 0;
  if (AudioObjectGetPropertyData(devId, &addr, 0, NULL, &dataSize, bufList) == noErr) {
    for (UInt32 i = 0; i < bufList->mNumberBuffers; i++) {
      total += bufList->mBuffers[i].mNumberChannels;
    }
  }
  free(bufList);
  return total;
}

static NSString *caDeviceName(AudioDeviceID devId) {
  AudioObjectPropertyAddress addr = {
    kAudioObjectPropertyName,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  CFStringRef cfName = NULL;
  UInt32 dataSize = sizeof(CFStringRef);
  if (AudioObjectGetPropertyData(devId, &addr, 0, NULL, &dataSize, &cfName) != noErr || !cfName) {
    return [NSString stringWithFormat:@"Audio Device %u", (unsigned)devId];
  }
  NSString *name = (__bridge_transfer NSString *)cfName;
  return name.length > 0 ? name : [NSString stringWithFormat:@"Audio Device %u", (unsigned)devId];
}

static int caCopyOut(NSString *s, char *out, int cap) {
  if (!out || cap <= 0) return 0;
  if (!s) {
    out[0] = 0;
    return 0;
  }
  const char *utf8 = [s UTF8String];
  if (!utf8) {
    out[0] = 0;
    return 0;
  }
  size_t len = strlen(utf8);
  if (len >= (size_t)cap) len = (size_t)cap - 1;
  memcpy(out, utf8, len);
  out[len] = 0;
  return (int)len;
}

int ov_audio_refresh_devices(void) {
  @autoreleasepool {
    os_unfair_lock_lock(&gDevLock);
    gInputDevices = [NSMutableArray array];
    gOutputDevices = [NSMutableArray array];
    [gInputDevices addObject:@{ @"id": @(0u), @"name": @"System Default Mic" }];
    [gOutputDevices addObject:@{ @"id": @(0u), @"name": @"System Default Output" }];

    AudioObjectPropertyAddress addr = {
      kAudioHardwarePropertyDevices,
      kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain
    };
    UInt32 dataSize = 0;
    BOOL foundSelIn = (gSelectedInputDeviceId == 0);
    BOOL foundSelOut = (gSelectedOutputDeviceId == 0);
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &dataSize) == noErr && dataSize > 0) {
      UInt32 count = dataSize / sizeof(AudioDeviceID);
      AudioDeviceID *ids = (AudioDeviceID *)malloc(dataSize);
      if (ids && AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &dataSize, ids) == noErr) {
        for (UInt32 i = 0; i < count; i++) {
          AudioDeviceID devId = ids[i];
          uint32_t inCh = caDeviceChannels(devId, kAudioDevicePropertyScopeInput);
          uint32_t outCh = caDeviceChannels(devId, kAudioDevicePropertyScopeOutput);
          if (inCh == 0 && outCh == 0) continue;
          NSString *name = caDeviceName(devId);
          if (inCh > 0) {
            [gInputDevices addObject:@{ @"id": @((uint32_t)devId), @"name": name }];
            if (devId == gSelectedInputDeviceId) foundSelIn = YES;
          }
          if (outCh > 0) {
            [gOutputDevices addObject:@{ @"id": @((uint32_t)devId), @"name": name }];
            if (devId == gSelectedOutputDeviceId) foundSelOut = YES;
          }
        }
      }
      if (ids) free(ids);
    }
    if (!foundSelIn) gSelectedInputDeviceId = 0;
    if (!foundSelOut) {
      gSelectedOutputDeviceId = 0;
      gPlayDeviceDirty = YES;
    }
    int total = (int)(gInputDevices.count + gOutputDevices.count);
    os_unfair_lock_unlock(&gDevLock);
    return total;
  }
}

int ov_audio_input_count(void) {
  os_unfair_lock_lock(&gDevLock);
  int n = gInputDevices ? (int)gInputDevices.count : 0;
  os_unfair_lock_unlock(&gDevLock);
  return n;
}

uint32_t ov_audio_input_id(int idx) {
  os_unfair_lock_lock(&gDevLock);
  uint32_t res = 0;
  if (gInputDevices && idx >= 0 && idx < (int)gInputDevices.count) {
    res = [gInputDevices[(NSUInteger)idx][@"id"] unsignedIntValue];
  }
  os_unfair_lock_unlock(&gDevLock);
  return res;
}

int ov_audio_input_name(int idx, char *out, int cap) {
  @autoreleasepool {
    os_unfair_lock_lock(&gDevLock);
    int len = 0;
    if (gInputDevices && idx >= 0 && idx < (int)gInputDevices.count) {
      len = caCopyOut(gInputDevices[(NSUInteger)idx][@"name"], out, cap);
    }
    os_unfair_lock_unlock(&gDevLock);
    return len;
  }
}

int ov_audio_output_count(void) {
  os_unfair_lock_lock(&gDevLock);
  int n = gOutputDevices ? (int)gOutputDevices.count : 0;
  os_unfair_lock_unlock(&gDevLock);
  return n;
}

uint32_t ov_audio_output_id(int idx) {
  os_unfair_lock_lock(&gDevLock);
  uint32_t res = 0;
  if (gOutputDevices && idx >= 0 && idx < (int)gOutputDevices.count) {
    res = [gOutputDevices[(NSUInteger)idx][@"id"] unsignedIntValue];
  }
  os_unfair_lock_unlock(&gDevLock);
  return res;
}

int ov_audio_output_name(int idx, char *out, int cap) {
  @autoreleasepool {
    os_unfair_lock_lock(&gDevLock);
    int len = 0;
    if (gOutputDevices && idx >= 0 && idx < (int)gOutputDevices.count) {
      len = caCopyOut(gOutputDevices[(NSUInteger)idx][@"name"], out, cap);
    }
    os_unfair_lock_unlock(&gDevLock);
    return len;
  }
}

uint32_t ov_audio_get_input_device(void) { return (uint32_t)gSelectedInputDeviceId; }
uint32_t ov_audio_get_output_device(void) { return (uint32_t)gSelectedOutputDeviceId; }

int ov_mic_start_rate(double preferredRate);
int ov_mic_start(void);
void ov_mic_stop(void);

int ov_audio_set_input_device(uint32_t devId) {
  gSelectedInputDeviceId = (AudioDeviceID)devId;
  return 1;
}

int ov_audio_set_output_device(uint32_t devId) {
  gSelectedOutputDeviceId = (AudioDeviceID)devId;
  gPlayDeviceDirty = YES;
  return 1;
}

// ============================================================================
// Microphone capture via CoreAudio AudioQueueNewInput
// (Zero ObjC NSExceptions, rock-solid device UID selection, automatic hardware
// sample-rate & mono float32 conversion — never touches the output device!)
// ============================================================================

#define MIC_QUEUE_CAP 262144  // ~5.4 s @ 48 kHz ring buffer between CoreAudio & UI poll
#define MIC_AQ_BUFFERS 4

static AudioQueueRef gRecQueue = NULL;
static AudioQueueBufferRef gRecBuffers[MIC_AQ_BUFFERS] = {0};
static os_unfair_lock gMicLock = OS_UNFAIR_LOCK_INIT;
static float gMicQueue[MIC_QUEUE_CAP];
static uint64_t gMicWrite = 0;
static uint64_t gMicRead = 0;
static double gMicRate = 48000.0;
static float gMicLevelDb = -120.0f;
static BOOL gMicRunning = NO;
static char gMicErr[256] = {0};

static void setMicError(NSString *msg) {
  os_unfair_lock_lock(&gMicLock);
  const char *s = msg ? msg.UTF8String : "";
  strncpy(gMicErr, s, sizeof(gMicErr) - 1);
  gMicErr[sizeof(gMicErr) - 1] = 0;
  os_unfair_lock_unlock(&gMicLock);
}

static AudioDeviceID caDefaultInputDevice(void) {
  AudioObjectPropertyAddress addr = {
    kAudioHardwarePropertyDefaultInputDevice,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  AudioDeviceID devId = kAudioObjectUnknown;
  UInt32 sz = sizeof(devId);
  if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &sz, &devId) != noErr) {
    return kAudioObjectUnknown;
  }
  return devId;
}

static double caDeviceNominalRate(AudioDeviceID devId) {
  if (devId == kAudioObjectUnknown) return 48000.0;
  AudioObjectPropertyAddress addr = {
    kAudioDevicePropertyNominalSampleRate,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  Float64 rate = 48000.0;
  UInt32 sz = sizeof(rate);
  if (AudioObjectGetPropertyData(devId, &addr, 0, NULL, &sz, &rate) == noErr && rate >= 8000.0 && rate <= 192000.0) {
    return (double)rate;
  }
  return 48000.0;
}

static CFStringRef caCopyDeviceUID(AudioDeviceID devId) {
  if (devId == kAudioObjectUnknown) return NULL;
  AudioObjectPropertyAddress addr = {
    kAudioDevicePropertyDeviceUID,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  CFStringRef uid = NULL;
  UInt32 sz = sizeof(uid);
  if (AudioObjectGetPropertyData(devId, &addr, 0, NULL, &sz, &uid) == noErr) {
    return uid;
  }
  return NULL;
}

static void ovMicAQCallback(
  void *inUserData,
  AudioQueueRef inAQ,
  AudioQueueBufferRef inBuffer,
  const AudioTimeStamp *inStartTime,
  UInt32 inNumberPacketDescriptions,
  const AudioStreamPacketDescription *inPacketDescs
) {
  (void)inUserData;
  (void)inStartTime;
  (void)inNumberPacketDescriptions;
  (void)inPacketDescs;
  if (!inBuffer || !inBuffer->mAudioData || inBuffer->mAudioDataByteSize == 0) {
    if (gMicRunning && inAQ) AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL);
    return;
  }

  const float *samples = (const float *)inBuffer->mAudioData;
  UInt32 frames = inBuffer->mAudioDataByteSize / sizeof(float);
  if (frames > 0) {
    double sumSq = 0.0;
    os_unfair_lock_lock(&gMicLock);
    for (UInt32 i = 0; i < frames; i++) {
      float s = samples[i];
      if (!isfinite(s)) s = 0.0f;
      sumSq += (double)s * (double)s;
      if (gMicWrite - gMicRead < MIC_QUEUE_CAP) {
        gMicQueue[gMicWrite % MIC_QUEUE_CAP] = s;
        gMicWrite++;
      }
    }
    double rms = sqrt(sumSq / (double)frames);
    gMicLevelDb = rms > 1e-6 ? (float)(20.0 * log10(rms)) : -120.0f;
    os_unfair_lock_unlock(&gMicLock);
  }

  if (gMicRunning && inAQ) {
    AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL);
  }
}

int ov_mic_start_rate(double preferredRate) {
  @autoreleasepool {
    if (gMicRunning) return 1;
    setMicError(@"");

    AVAuthorizationStatus auth = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    if (auth == AVAuthorizationStatusDenied || auth == AVAuthorizationStatusRestricted) {
      setMicError(@"Microphone permission denied in macOS System Settings → Privacy & Security → Microphone.");
      return 0;
    }
    if (auth == AVAuthorizationStatusNotDetermined) {
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      __block BOOL granted = NO;
      [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL ok) {
        granted = ok;
        dispatch_semaphore_signal(sem);
      }];
      dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC)));
      if (!granted) {
        setMicError(@"Microphone permission was not granted.");
        return 0;
      }
    }

    if (gRecQueue) {
      AudioQueueStop(gRecQueue, true);
      AudioQueueDispose(gRecQueue, true);
      gRecQueue = NULL;
    }

    AudioDeviceID devId = (gSelectedInputDeviceId != 0) ? gSelectedInputDeviceId : caDefaultInputDevice();
    double targetRate = (preferredRate >= 8000.0 && preferredRate <= 192000.0)
      ? preferredRate
      : caDeviceNominalRate(devId);
    if (targetRate < 8000.0 || targetRate > 192000.0) targetRate = 48000.0;

    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.mSampleRate = targetRate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mBytesPerPacket = sizeof(float);
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = sizeof(float);
    fmt.mChannelsPerFrame = 1;
    fmt.mBitsPerChannel = 32;

    OSStatus st = AudioQueueNewInput(&fmt, ovMicAQCallback, NULL, NULL, NULL, 0, &gRecQueue);
    if (st != noErr || !gRecQueue) {
      setMicError([NSString stringWithFormat:@"Could not open audio input queue (OSStatus %d).", (int)st]);
      gRecQueue = NULL;
      return 0;
    }

    // Bind explicit selected microphone device if user chose a specific input
    if (gSelectedInputDeviceId != 0) {
      CFStringRef uid = caCopyDeviceUID(gSelectedInputDeviceId);
      if (uid) {
        OSStatus devSt = AudioQueueSetProperty(gRecQueue, kAudioQueueProperty_CurrentDevice, &uid, sizeof(uid));
        CFRelease(uid);
        if (devSt != noErr) {
          gSelectedInputDeviceId = 0;
        }
      }
    }

    os_unfair_lock_lock(&gMicLock);
    gMicWrite = 0;
    gMicRead = 0;
    gMicRate = targetRate;
    gMicLevelDb = -120.0f;
    os_unfair_lock_unlock(&gMicLock);

    UInt32 bufFrames = 1024;
    UInt32 bufBytes = bufFrames * sizeof(float);
    for (int i = 0; i < MIC_AQ_BUFFERS; i++) {
      gRecBuffers[i] = NULL;
      if (AudioQueueAllocateBuffer(gRecQueue, bufBytes, &gRecBuffers[i]) != noErr) {
        AudioQueueDispose(gRecQueue, true);
        gRecQueue = NULL;
        setMicError(@"Failed to allocate CoreAudio input buffers.");
        return 0;
      }
      AudioQueueEnqueueBuffer(gRecQueue, gRecBuffers[i], 0, NULL);
    }

    gMicRunning = YES;
    st = AudioQueueStart(gRecQueue, NULL);
    if (st != noErr) {
      gMicRunning = NO;
      AudioQueueStop(gRecQueue, true);
      AudioQueueDispose(gRecQueue, true);
      gRecQueue = NULL;
      setMicError([NSString stringWithFormat:@"Failed to start microphone capture (OSStatus %d).", (int)st]);
      return 0;
    }
    return 1;
  }
}

int ov_mic_start(void) {
  return ov_mic_start_rate(0.0);
}

void ov_mic_stop(void) {
  @autoreleasepool {
    gMicRunning = NO;
    if (gRecQueue) {
      AudioQueueStop(gRecQueue, true);
      AudioQueueDispose(gRecQueue, true);
      gRecQueue = NULL;
      for (int i = 0; i < MIC_AQ_BUFFERS; i++) {
        gRecBuffers[i] = NULL;
      }
    }
    os_unfair_lock_lock(&gMicLock);
    gMicLevelDb = -120.0f;
    os_unfair_lock_unlock(&gMicLock);
  }
}

int ov_mic_running(void) { return gMicRunning ? 1 : 0; }
double ov_mic_sample_rate(void) {
  os_unfair_lock_lock(&gMicLock);
  double r = gMicRate;
  os_unfair_lock_unlock(&gMicLock);
  return r;
}
float ov_mic_level_db(void) {
  os_unfair_lock_lock(&gMicLock);
  float db = gMicLevelDb;
  os_unfair_lock_unlock(&gMicLock);
  return db;
}

int ov_mic_available(void) {
  os_unfair_lock_lock(&gMicLock);
  uint64_t avail = gMicWrite - gMicRead;
  os_unfair_lock_unlock(&gMicLock);
  return avail > 0x7fffffff ? 0x7fffffff : (int)avail;
}

int ov_mic_drain(float *out, int maxCount) {
  if (!out || maxCount <= 0) return 0;
  os_unfair_lock_lock(&gMicLock);
  uint64_t avail = gMicWrite - gMicRead;
  int n = (int)((uint64_t)maxCount < avail ? (uint64_t)maxCount : avail);
  for (int i = 0; i < n; i++) {
    out[i] = gMicQueue[(gMicRead + (uint64_t)i) % MIC_QUEUE_CAP];
  }
  gMicRead += (uint64_t)n;
  os_unfair_lock_unlock(&gMicLock);
  return n;
}

int ov_mic_error(char *out, int capacity) {
  if (!out || capacity <= 0) return 0;
  os_unfair_lock_lock(&gMicLock);
  int len = (int)strlen(gMicErr);
  if (len > capacity - 1) len = capacity - 1;
  memcpy(out, gMicErr, (size_t)len);
  out[len] = 0;
  os_unfair_lock_unlock(&gMicLock);
  return len;
}

// ============================================================================
// Audio file decoding (WAV, MP3, M4A, AAC, FLAC, AIFF, CAF -> mono f32)
// ============================================================================

#define MAX_LOADED_SECONDS 600.0

static float *gLoadedSamples = NULL;
static int gLoadedCount = 0;
static double gLoadedRate = 48000.0;
static char gLoadedName[512] = {0};

int ov_load_file(const char *path, size_t length) {
  @autoreleasepool {
    if (!path || length == 0) return 0;
    NSString *nsPath = [[NSString alloc] initWithBytes:path length:length encoding:NSUTF8StringEncoding];
    if (!nsPath) return 0;
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    NSError *err = nil;
    AVAudioFile *file = [[AVAudioFile alloc] initForReading:url error:&err];
    if (!file || file.length <= 0) return 0;

    AVAudioFormat *proc = file.processingFormat;
    double rate = proc.sampleRate;
    AVAudioFramePosition maxFrames = (AVAudioFramePosition)(MAX_LOADED_SECONDS * rate);
    AVAudioFramePosition totalFrames = file.length;
    if (totalFrames > maxFrames) totalFrames = maxFrames;

    float *mono = (float *)malloc((size_t)totalFrames * sizeof(float));
    if (!mono) return 0;

    const AVAudioFrameCount chunk = 65536;
    AVAudioPCMBuffer *buf = [[AVAudioPCMBuffer alloc] initWithPCMFormat:proc frameCapacity:chunk];
    AVAudioFramePosition written = 0;

    while (written < totalFrames) {
      AVAudioFrameCount want = (AVAudioFrameCount)((totalFrames - written) < chunk ? (totalFrames - written) : chunk);
      buf.frameLength = 0;
      NSError *readErr = nil;
      if (![file readIntoBuffer:buf frameCount:want error:&readErr] || buf.frameLength == 0) {
        break;
      }
      float *const *ch = buf.floatChannelData;
      AVAudioChannelCount channels = proc.channelCount;
      AVAudioFrameCount got = buf.frameLength;
      if (channels == 1) {
        memcpy(mono + written, ch[0], (size_t)got * sizeof(float));
      } else {
        float inv = 1.0f / (float)channels;
        for (AVAudioFrameCount i = 0; i < got; i++) {
          float sum = 0.0f;
          for (AVAudioChannelCount c = 0; c < channels; c++) sum += ch[c][i];
          mono[written + i] = sum * inv;
        }
      }
      written += got;
    }

    if (gLoadedSamples) free(gLoadedSamples);
    gLoadedSamples = mono;
    gLoadedCount = (int)written;
    gLoadedRate = rate;

    NSString *base = url.lastPathComponent ?: @"Audio";
    const char *utf8 = base.UTF8String;
    strncpy(gLoadedName, utf8 ? utf8 : "Audio", sizeof(gLoadedName) - 1);
    gLoadedName[sizeof(gLoadedName) - 1] = 0;
    return written > 0 ? 1 : 0;
  }
}

int ov_loaded_count(void) { return gLoadedCount; }
double ov_loaded_rate(void) { return gLoadedRate; }

int ov_loaded_copy(float *out, int maxCount) {
  if (!out || !gLoadedSamples || maxCount <= 0 || gLoadedCount <= 0) return 0;
  int n = maxCount < gLoadedCount ? maxCount : gLoadedCount;
  memcpy(out, gLoadedSamples, (size_t)n * sizeof(float));
  return n;
}

int ov_loaded_name(char *out, int capacity) {
  if (!out || capacity <= 0) return 0;
  int len = (int)strlen(gLoadedName);
  if (len > capacity - 1) len = capacity - 1;
  memcpy(out, gLoadedName, (size_t)len);
  out[len] = 0;
  return len;
}

// ============================================================================
// Audio playback (mono f32 buffer with optional range & loop)
// ============================================================================

static AVAudioEngine *gPlayEngine = nil;
static AVAudioPlayerNode *gPlayer = nil;
static AVAudioUnitTimePitch *gTimePitch = nil;
static AVAudioSourceNode *gScrubSource = nil;
static AVAudioMixerNode *gFader = nil;
static double gPlayRate = 48000.0;
static double gPlayStartSec = 0.0;
static double gPlayEndSec = 0.0;
static double gPlayLastPos = 0.0;
static BOOL gPlayLooping = NO;
static BOOL gPlayActive = NO;
static float gPlayVolume = 0.85f;
static float gPlaySpeed = 1.0f;
static double gPlayWallStart = 0.0;
static double gPlayExpectedWallDur = 0.0;
static atomic_int gPlayGen = 0;
static atomic_int gPlayFinished = 0;

// Continuous WSOLA Freeze-Hold Granular Scrubbing State
static os_unfair_lock gScrubLock = OS_UNFAIR_LOCK_INIT;
static float *gScrubBuf = NULL;
static int gScrubCount = 0;
static double gScrubRate = 48000.0;
static double gScrubTargetSec = 0.0;
static double gScrubSmoothSec = 0.0;
static BOOL gScrubActive = NO;
static float gScrubEnv = 0.0f;
static int gGrainLen = 3840;
static int gGrainAStart = 0;
static int gGrainAPos = 0;
static int gGrainBStart = 0;
static int gGrainBPos = 1920;

static int pickAlignedGrainStart(const float *buf, int count, int targetStart, int refPos, int grainLen, int searchRad) {
  int maxStart = count - grainLen;
  if (maxStart <= 0) return 0;
  if (targetStart < 0) targetStart = 0;
  if (targetStart > maxStart) targetStart = maxStart;
  if (refPos < 0 || refPos + 48 >= count) return targetStart;

  int lo = targetStart - searchRad;
  if (lo < 0) lo = 0;
  int hi = targetStart + searchRad;
  if (hi > maxStart) hi = maxStart;

  // WSOLA micro-search: align new grain phase to the currently sounding grain so 50%-overlap Hann sums cleanly!
  int best = targetStart;
  double bestScore = -1e30;
  for (int cand = lo; cand <= hi; cand += 2) {
    double dot = 0.0;
    for (int k = 0; k < 48; k += 2) {
      dot += (double)buf[cand + k] * (double)buf[refPos + k];
    }
    if (dot > bestScore) {
      bestScore = dot;
      best = cand;
    }
  }
  return best;
}

static void ensurePlayEngine(double sampleRate) {
  if (gPlayEngine && !gPlayDeviceDirty && fabs(gPlayRate - sampleRate) < 1.0) return;
  gPlayDeviceDirty = NO;
  if (gPlayEngine) {
    [gPlayer stop];
    [gPlayEngine stop];
    gPlayEngine = nil;
    gPlayer = nil;
    gTimePitch = nil;
    gScrubSource = nil;
    gFader = nil;
  }
  gPlayRate = sampleRate > 0 ? sampleRate : 48000.0;
  gPlayEngine = [[AVAudioEngine alloc] init];
  if (gSelectedOutputDeviceId != 0) {
    AudioUnit outAU = gPlayEngine.outputNode.audioUnit;
    if (outAU) {
      AudioDeviceID devId = gSelectedOutputDeviceId;
      OSStatus st = AudioUnitSetProperty(
        outAU,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &devId,
        sizeof(devId)
      );
      if (st != noErr) {
        gSelectedOutputDeviceId = 0;
      }
    }
  }
  gPlayer = [[AVAudioPlayerNode alloc] init];
  gTimePitch = [[AVAudioUnitTimePitch alloc] init];
  gTimePitch.rate = gPlaySpeed;
  gTimePitch.pitch = 0.0f;

  AVAudioFormat *monoFmt = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:gPlayRate channels:1];
  double engineRate = gPlayRate;
  gScrubSource = [[AVAudioSourceNode alloc] initWithFormat:monoFmt renderBlock:^OSStatus(
    BOOL *isSilence,
    const AudioTimeStamp *timestamp,
    AVAudioFrameCount frameCount,
    AudioBufferList *outputData
  ) {
    (void)timestamp;
    float *out = (float *)outputData->mBuffers[0].mData;
    if (!out || frameCount == 0) return noErr;

    os_unfair_lock_lock(&gScrubLock);
    if ((!gScrubActive && gScrubEnv <= 0.0001f) || !gScrubBuf || gScrubCount <= 0) {
      gScrubEnv = 0.0f;
      os_unfair_lock_unlock(&gScrubLock);
      memset(out, 0, frameCount * sizeof(float));
      *isSilence = YES;
      return noErr;
    }

    int gLen = gGrainLen;
    if (gLen < 512) gLen = 512;
    int halfLen = gLen / 2;
    double inv = 2.0 * M_PI / (double)(gLen - 1);
    float envStepUp = 1.0f / (float)fmax(64.0, engineRate * 0.008);
    float envStepDn = 1.0f / (float)fmax(128.0, engineRate * 0.018);
    int searchRad = (int)round(gScrubRate * 0.009);
    if (searchRad < 64) searchRad = 64;
    if (searchRad > 600) searchRad = 600;

    for (AVAudioFrameCount i = 0; i < frameCount; i++) {
      // Smooth cursor gliding (~8 ms) so slow drags morph vowels continuously with zero jumps
      gScrubSmoothSec += (gScrubTargetSec - gScrubSmoothSec) * 0.0025;
      int centerIdx = (int)round(gScrubSmoothSec * gScrubRate);
      int targetStart = centerIdx - halfLen;

      if (gGrainAPos >= gLen) {
        gGrainAPos = 0;
        int refIdx = gGrainBStart + gGrainBPos;
        gGrainAStart = pickAlignedGrainStart(gScrubBuf, gScrubCount, targetStart, refIdx, gLen, searchRad);
      }
      if (gGrainBPos >= gLen) {
        gGrainBPos = 0;
        int refIdx = gGrainAStart + gGrainAPos;
        gGrainBStart = pickAlignedGrainStart(gScrubBuf, gScrubCount, targetStart, refIdx, gLen, searchRad);
      }

      int idxA = gGrainAStart + gGrainAPos;
      int idxB = gGrainBStart + gGrainBPos;
      float sA = (idxA >= 0 && idxA < gScrubCount) ? gScrubBuf[idxA] : 0.0f;
      float sB = (idxB >= 0 && idxB < gScrubCount) ? gScrubBuf[idxB] : 0.0f;
      float wA = (float)(0.5 - 0.5 * cos(inv * (double)gGrainAPos));
      float wB = (float)(0.5 - 0.5 * cos(inv * (double)gGrainBPos));

      if (gScrubActive) {
        gScrubEnv += envStepUp;
        if (gScrubEnv > 1.0f) gScrubEnv = 1.0f;
      } else {
        gScrubEnv -= envStepDn;
        if (gScrubEnv < 0.0f) gScrubEnv = 0.0f;
      }

      out[i] = (sA * wA + sB * wB) * gScrubEnv;
      gGrainAPos++;
      gGrainBPos++;
    }
    os_unfair_lock_unlock(&gScrubLock);
    *isSilence = NO;
    return noErr;
  }];

  gFader = [[AVAudioMixerNode alloc] init];
  gFader.outputVolume = gPlayVolume;
  [gPlayEngine attachNode:gPlayer];
  [gPlayEngine attachNode:gTimePitch];
  [gPlayEngine attachNode:gScrubSource];
  [gPlayEngine attachNode:gFader];

  [gPlayEngine connect:gPlayer to:gTimePitch format:monoFmt];
  [gPlayEngine connect:gTimePitch to:gFader format:monoFmt];
  [gPlayEngine connect:gScrubSource to:gFader format:monoFmt];
  [gPlayEngine connect:gFader to:gPlayEngine.mainMixerNode format:monoFmt];
  @try {
    [gPlayEngine startAndReturnError:nil];
  } @catch (NSException *ex) {
    (void)ex;
  }
}

void ov_play_set_speed(float speed) {
  if (speed < 0.25f) speed = 0.25f;
  if (speed > 2.0f) speed = 2.0f;
  if (gPlayActive && !gPlayLooping && gPlaySpeed > 0.0f) {
    double now = CFAbsoluteTimeGetCurrent();
    double elapsed = fmax(0.0, now - gPlayWallStart);
    double remaining = fmax(0.0, gPlayExpectedWallDur - elapsed);
    gPlayWallStart = now;
    gPlayExpectedWallDur = remaining * ((double)gPlaySpeed / (double)speed);
  }
  gPlaySpeed = speed;
  if (gTimePitch) {
    gTimePitch.rate = speed;
  }
}

float ov_play_get_speed(void) {
  return gPlaySpeed;
}

void ov_scrub_begin(const float *samples, int totalCount, double sampleRate, double centerSec) {
  @autoreleasepool {
    if (!samples || totalCount <= 0 || sampleRate <= 0) return;
    ensurePlayEngine(sampleRate);
    if (!gPlayEngine.isRunning) {
      @try {
        [gPlayEngine startAndReturnError:nil];
      } @catch (NSException *ex) {
        (void)ex;
      }
    }

    float *copy = (float *)malloc((size_t)totalCount * sizeof(float));
    if (!copy) return;
    memcpy(copy, samples, (size_t)totalCount * sizeof(float));

    int gLen = (int)round(sampleRate * 0.080);
    if (gLen < 512) gLen = 512;
    if (gLen > 8192) gLen = 8192;
    gLen = (gLen / 2) * 2;
    int start = (int)round(centerSec * sampleRate) - gLen / 2;
    if (start < 0) start = 0;
    if (start > totalCount - gLen) start = totalCount > gLen ? totalCount - gLen : 0;

    os_unfair_lock_lock(&gScrubLock);
    float *old = gScrubBuf;
    gScrubBuf = copy;
    gScrubCount = totalCount;
    gScrubRate = sampleRate;
    gScrubTargetSec = centerSec;
    gScrubSmoothSec = centerSec;
    gGrainLen = gLen;
    gGrainAStart = start;
    gGrainAPos = 0;
    gGrainBStart = start;
    gGrainBPos = gLen / 2;
    gScrubActive = YES;
    os_unfair_lock_unlock(&gScrubLock);
    if (old) free(old);
  }
}

void ov_scrub_move(double centerSec) {
  os_unfair_lock_lock(&gScrubLock);
  gScrubTargetSec = centerSec;
  os_unfair_lock_unlock(&gScrubLock);
}

void ov_scrub_stop(void) {
  os_unfair_lock_lock(&gScrubLock);
  gScrubActive = NO;
  os_unfair_lock_unlock(&gScrubLock);
}

int ov_play_range(const float *samples, int totalCount, double sampleRate, double startSec, double endSec, int loop) {
  @autoreleasepool {
    if (!samples || totalCount <= 0 || sampleRate <= 0) return 0;
    // If CoreAudio stopped the output engine (e.g. Bluetooth mic HFP<->A2DP route change during Drill),
    // rebuild the AVAudioEngine graph cleanly so playback never drops out or reports stale sample times.
    if (gPlayEngine && !gPlayEngine.isRunning) {
      gPlayDeviceDirty = YES;
    }
    ensurePlayEngine(sampleRate);
    if (!gPlayEngine.isRunning) {
      @try {
        [gPlayEngine startAndReturnError:nil];
      } @catch (NSException *ex) {
        (void)ex;
      }
    }

    int startIdx = (int)floor(startSec * sampleRate);
    int endIdx = (int)ceil(endSec * sampleRate);
    if (startIdx < 0) startIdx = 0;
    if (endIdx > totalCount) endIdx = totalCount;
    if (endIdx <= startIdx) {
      startIdx = 0;
      endIdx = totalCount;
    }
    int count = endIdx - startIdx;
    if (count <= 0) return 0;

    AVAudioFormat *monoFmt = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:gPlayRate channels:1];
    AVAudioPCMBuffer *buf = [[AVAudioPCMBuffer alloc] initWithPCMFormat:monoFmt frameCapacity:(AVAudioFrameCount)count];
    buf.frameLength = (AVAudioFrameCount)count;
    memcpy(buf.floatChannelData[0], samples + startIdx, (size_t)count * sizeof(float));

    int gen = atomic_fetch_add(&gPlayGen, 1) + 1;
    atomic_store(&gPlayFinished, 0);
    [gPlayer stop];
    if (gTimePitch) {
      // Flush AVAudioUnitTimePitch internal delay/sample-time state so subsequent plays
      // (like Repeat Drill rep #2+) never fire CompletionDataPlayedBack prematurely!
      [gTimePitch reset];
      gTimePitch.rate = gPlaySpeed;
    }

    gPlayStartSec = (double)startIdx / sampleRate;
    gPlayEndSec = (double)endIdx / sampleRate;
    gPlayLastPos = gPlayStartSec;
    gPlayLooping = loop != 0;
    gPlayWallStart = CFAbsoluteTimeGetCurrent();
    gPlayExpectedWallDur = ((double)count / sampleRate) / (double)fmax(0.25f, gPlaySpeed);
    gPlayActive = YES;

    AVAudioPlayerNodeBufferOptions opts = gPlayLooping ? AVAudioPlayerNodeBufferLoops : 0;
    [gPlayer scheduleBuffer:buf
                     atTime:nil
                    options:opts
     completionCallbackType:AVAudioPlayerNodeCompletionDataPlayedBack
          completionHandler:^(AVAudioPlayerNodeCompletionCallbackType type) {
      (void)type;
      if (!gPlayLooping && atomic_load(&gPlayGen) == gen) {
        atomic_store(&gPlayFinished, 1);
      }
    }];
    [gPlayer play];
    return 1;
  }
}

void ov_play_stop(void) {
  if (!gPlayer) return;
  atomic_fetch_add(&gPlayGen, 1);
  atomic_store(&gPlayFinished, 0);
  [gPlayer stop];
  if (gTimePitch) {
    [gTimePitch reset];
  }
  gPlayActive = NO;
}

int ov_play_is_playing(void) {
  return (gPlayActive && gPlayer && gPlayer.isPlaying) ? 1 : 0;
}

int ov_play_take_finished(void) {
  if (!gPlayActive || gPlayLooping) {
    atomic_store(&gPlayFinished, 0);
    return 0;
  }
  double elapsed = CFAbsoluteTimeGetCurrent() - gPlayWallStart;
  int cbDone = atomic_load(&gPlayFinished);
  // Guard against macOS AVAudioPlayerNode + AVAudioUnitTimePitch premature completion callbacks
  // on 2nd+ playback (Repeat Drill rep #2+): only honor completion once the real wall duration
  // has nearly elapsed, and guarantee completion shortly after expected duration if callback stalls.
  BOOL doneByCallback = (cbDone != 0) && (elapsed >= gPlayExpectedWallDur - 0.12);
  BOOL doneByClock = (elapsed >= gPlayExpectedWallDur + 0.12);
  if (doneByCallback || doneByClock) {
    atomic_store(&gPlayFinished, 0);
    gPlayActive = NO;
    return 1;
  }
  return 0;
}

double ov_play_position(void) {
  if (!gPlayer || !gPlayActive || !gPlayer.isPlaying) return gPlayLastPos;
  AVAudioTime *nodeTime = gPlayer.lastRenderTime;
  if (nodeTime && nodeTime.isSampleTimeValid) {
    AVAudioTime *pt = [gPlayer playerTimeForNodeTime:nodeTime];
    if (pt && pt.isSampleTimeValid && pt.sampleRate > 0) {
      double elapsed = (double)pt.sampleTime / pt.sampleRate;
      double span = gPlayEndSec - gPlayStartSec;
      if (span > 0.0) {
        if (gPlayLooping) {
          double mod = fmod(elapsed, span);
          if (mod < 0.0) mod += span;
          gPlayLastPos = gPlayStartSec + mod;
        } else {
          double pos = gPlayStartSec + elapsed;
          if (pos > gPlayEndSec) pos = gPlayEndSec;
          if (pos < gPlayStartSec) pos = gPlayStartSec;
          gPlayLastPos = pos;
        }
      }
    }
  }
  return gPlayLastPos;
}

void ov_play_set_volume(float volume) {
  gPlayVolume = volume < 0.0f ? 0.0f : (volume > 1.0f ? 1.0f : volume);
  if (gFader) gFader.outputVolume = gPlayVolume;
}

// ============================================================================
// 16-bit PCM WAV Export
// ============================================================================

static void write_u16_le(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
}

static void write_u32_le(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
  p[2] = (uint8_t)((v >> 16) & 0xff);
  p[3] = (uint8_t)((v >> 24) & 0xff);
}

int ov_export_wav(const char *path, size_t pathLen, const float *samples, int count, int sampleRate) {
  @autoreleasepool {
    if (!path || pathLen == 0 || !samples || count <= 0 || sampleRate <= 0) return 0;
    char *cpath = (char *)malloc(pathLen + 1);
    if (!cpath) return 0;
    memcpy(cpath, path, pathLen);
    cpath[pathLen] = 0;
    FILE *f = fopen(cpath, "wb");
    free(cpath);
    if (!f) return 0;

    uint32_t dataBytes = (uint32_t)count * 2;
    uint8_t hdr[44];
    memcpy(hdr + 0, "RIFF", 4);
    write_u32_le(hdr + 4, 36 + dataBytes);
    memcpy(hdr + 8, "WAVE", 4);
    memcpy(hdr + 12, "fmt ", 4);
    write_u32_le(hdr + 16, 16);
    write_u16_le(hdr + 20, 1);  // PCM
    write_u16_le(hdr + 22, 1);  // mono
    write_u32_le(hdr + 24, (uint32_t)sampleRate);
    write_u32_le(hdr + 28, (uint32_t)sampleRate * 2);
    write_u16_le(hdr + 32, 2);
    write_u16_le(hdr + 34, 16);
    memcpy(hdr + 36, "data", 4);
    write_u32_le(hdr + 40, dataBytes);
    fwrite(hdr, 1, 44, f);

    int16_t chunk[4096];
    int pos = 0;
    while (pos < count) {
      int n = (count - pos) < 4096 ? (count - pos) : 4096;
      for (int i = 0; i < n; i++) {
        float v = samples[pos + i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        chunk[i] = (int16_t)lrintf(v * 32767.0f);
      }
      fwrite(chunk, sizeof(int16_t), (size_t)n, f);
      pos += n;
    }
    fclose(f);
    return 1;
  }
}

// ============================================================================
// Accelerate vDSP FFT helpers (real magnitudes & autocorrelation)
// ============================================================================

#define MAX_LOG2 15  // up to 32768-point FFT

static FFTSetup gFftSetups[MAX_LOG2 + 1] = {0};
static os_unfair_lock gFftLock = OS_UNFAIR_LOCK_INIT;

static FFTSetup getFftSetup(int log2n) {
  if (log2n < 2 || log2n > MAX_LOG2) return NULL;
  os_unfair_lock_lock(&gFftLock);
  if (!gFftSetups[log2n]) {
    gFftSetups[log2n] = vDSP_create_fftsetup((vDSP_Length)log2n, kFFTRadix2);
  }
  FFTSetup s = gFftSetups[log2n];
  os_unfair_lock_unlock(&gFftLock);
  return s;
}

static int ilog2_pow2(int n) {
  if (n < 4 || (n & (n - 1)) != 0) return -1;
  int p = 0;
  while ((1 << p) < n) p++;
  return p;
}

// Computes column-major dB spectrogram columns matching voiceapp/columns.ts bit-for-bit:
//   - Windows centred at `firstCentre + c * hop`
//   - Multiplied by `window[0..windowSize)`
//   - Normalized by `coherentGain` so a full-scale sine reads 0 dBFS
//   - Clamped to `dbFloor`
//   - Stores the first `outBins` bins per column into `outDb[c * outBins + b]`
void ov_compute_columns_db(
  const float *samples,
  int sampleCount,
  int firstCentre,
  int hop,
  int numColumns,
  const float *window,
  int windowSize,
  float coherentGain,
  float dbFloor,
  int outBins,
  float *outDb
) {
  int log2n = ilog2_pow2(windowSize);
  if (log2n < 2 || log2n > MAX_LOG2 || numColumns <= 0 || outBins <= 0) return;
  FFTSetup setup = getFftSetup(log2n);
  if (!setup) return;

  int half = windowSize >> 1;
  if (outBins > half) outBins = half;

  float *frame = (float *)malloc((size_t)windowSize * sizeof(float));
  float *re = (float *)malloc((size_t)half * sizeof(float));
  float *im = (float *)malloc((size_t)half * sizeof(float));
  if (!frame || !re || !im) {
    free(frame);
    free(re);
    free(im);
    return;
  }

  DSPSplitComplex split = {re, im};
  float invN = 1.0f / (float)windowSize;
  float invGain = coherentGain > 0.0f ? (1.0f / coherentGain) : 1.0f;
  float magScale = invN * invGain;
  float dcScale = 0.5f * invN * invGain;
  float magFloor = powf(10.0f, dbFloor / 20.0f);

  for (int c = 0; c < numColumns; c++) {
    int centre = firstCentre + c * hop;
    int start = centre - half;

    // Fast path when fully inside samples
    if (start >= 0 && start + windowSize <= sampleCount) {
      vDSP_vmul(samples + start, 1, window, 1, frame, 1, (vDSP_Length)windowSize);
    } else {
      for (int i = 0; i < windowSize; i++) {
        int idx = start + i;
        float s = (idx >= 0 && idx < sampleCount) ? samples[idx] : 0.0f;
        frame[i] = s * window[i];
      }
    }

    vDSP_ctoz((const DSPComplex *)frame, 2, &split, 1, (vDSP_Length)half);
    vDSP_fft_zrip(setup, &split, 1, (vDSP_Length)log2n, kFFTDirection_Forward);

    float *colOut = outDb + (size_t)c * (size_t)outBins;
    float dcMag = fabsf(re[0]) * dcScale;
    colOut[0] = dcMag <= magFloor ? dbFloor : fmaxf(dbFloor, 20.0f * log10f(dcMag));

    for (int b = 1; b < outBins; b++) {
      float m = hypotf(re[b], im[b]) * magScale;
      colOut[b] = m <= magFloor ? dbFloor : fmaxf(dbFloor, 20.0f * log10f(m));
    }
  }

  free(frame);
  free(re);
  free(im);
}

// Fast -O3 RGBA8 texture packer for the WebGPU spectrogram (row 0 = highest bin, last row = DC).
void ov_columns_to_rgba(
  const float *data,
  int colOffset,
  int cols,
  int bins,
  float floorDb,
  float ceilingDb,
  uint8_t *rgba
) {
  if (!data || !rgba || cols <= 0 || bins <= 0) return;
  float range = ceilingDb - floorDb;
  if (range < 1.0f) range = 1.0f;
  float invRange = 255.0f / range;
  for (int b = 0; b < bins; b++) {
    int dstRow = (bins - 1 - b) * cols;
    for (int c = 0; c < cols; c++) {
      float db = data[(colOffset + c) * bins + b];
      float scaled = (db - floorDb) * invRange + 0.5f;
      if (scaled < 0.0f) scaled = 0.0f;
      if (scaled > 255.0f) scaled = 255.0f;
      uint8_t v = (uint8_t)scaled;
      int px = (dstRow + c) * 4;
      rgba[px + 0] = v;
      rgba[px + 1] = v;
      rgba[px + 2] = v;
      rgba[px + 3] = 255;
    }
  }
}

// Autocorrelation via vDSP real FFT (Wiener-Khinchin) for Boersma (1993) pitch detection.
// `input` has length `inputLen`, zero-padded into `fftSize` (power of 2 >= inputLen * 3/2).
// Writes lags `0..outLen-1` into `outAcf` (double precision).
void ov_autocorr_fft(const float *input, int inputLen, int fftSize, double *outAcf, int outLen) {
  int log2n = ilog2_pow2(fftSize);
  if (log2n < 2 || log2n > MAX_LOG2 || inputLen <= 0 || outLen <= 0) return;
  FFTSetup setup = getFftSetup(log2n);
  if (!setup) return;

  int half = fftSize >> 1;
  float *buf = (float *)calloc((size_t)fftSize, sizeof(float));
  float *re = (float *)malloc((size_t)half * sizeof(float));
  float *im = (float *)malloc((size_t)half * sizeof(float));
  if (!buf || !re || !im) {
    free(buf);
    free(re);
    free(im);
    return;
  }

  int copyLen = inputLen < fftSize ? inputLen : fftSize;
  memcpy(buf, input, (size_t)copyLen * sizeof(float));

  DSPSplitComplex split = {re, im};
  vDSP_ctoz((const DSPComplex *)buf, 2, &split, 1, (vDSP_Length)half);
  vDSP_fft_zrip(setup, &split, 1, (vDSP_Length)log2n, kFFTDirection_Forward);

  // Power spectrum: packed DC in re[0], Nyquist in im[0], complex bins 1..half-1
  re[0] = re[0] * re[0];
  im[0] = im[0] * im[0];
  for (int k = 1; k < half; k++) {
    re[k] = re[k] * re[k] + im[k] * im[k];
    im[k] = 0.0f;
  }

  vDSP_fft_zrip(setup, &split, 1, (vDSP_Length)log2n, kFFTDirection_Inverse);
  vDSP_ztoc(&split, 1, (DSPComplex *)buf, 2, (vDSP_Length)half);

  // vDSP forward multiplies by 2, squaring multiplies by 4, inverse multiplies by N -> scale by 1/(4*N)
  double scale = 1.0 / (4.0 * (double)fftSize);
  int n = outLen < fftSize ? outLen : fftSize;
  for (int i = 0; i < n; i++) {
    outAcf[i] = (double)buf[i] * scale;
  }

  free(buf);
  free(re);
  free(im);
}

// Fast native windowed-sinc anti-aliased resampling + Praat +6 dB/oct pre-emphasis for Burg LPC.
// Returns pre-preemphasis RMS intensity in dBFS.
double ov_resample_preemph(
  const float *src,
  int srcLen,
  double srcRate,
  double centreSec,
  double dstRate,
  double preemphHz,
  double *dst,
  int n
) {
  if (!src || srcLen <= 0 || !dst || n <= 0 || srcRate <= 0.0 || dstRate <= 0.0) {
    return -120.0;
  }
  double *work = (double *)malloc((size_t)(n + 1) * sizeof(double));
  if (!work) return -120.0;

  double ratio = srcRate / dstRate;
  double cutoff = ratio > 1.0 ? (0.92 / ratio) : 1.0;
  int halfTaps = ratio > 1.0 ? (int)ceil(6.0 * ratio) : 8;
  if (halfTaps > 36) halfTaps = 36;
  double halfTapsF = (double)halfTaps;
  double invWinDenom = M_PI / (halfTapsF + 1.0);
  double startTime = centreSec - ((double)n * 0.5 + 1.0) / dstRate;

  for (int k = 0; k <= n; k++) {
    double t = startTime + (double)k / dstRate;
    double center = t * srcRate;
    if (center < -halfTapsF || center > (double)srcLen + halfTapsF) {
      work[k] = 0.0;
    } else if (ratio <= 1.01) {
      int i0 = (int)floor(center);
      double frac = center - (double)i0;
      double s0 = (i0 >= 0 && i0 < srcLen) ? (double)src[i0] : 0.0;
      double s1 = (i0 + 1 >= 0 && i0 + 1 < srcLen) ? (double)src[i0 + 1] : 0.0;
      work[k] = s0 + (s1 - s0) * frac;
    } else {
      int cInt = (int)llround(center);
      int lo = cInt - halfTaps;
      int hi = cInt + halfTaps;
      if (lo < 0) lo = 0;
      if (hi >= srcLen) hi = srcLen - 1;
      double acc = 0.0;
      double wsum = 0.0;
      for (int j = lo; j <= hi; j++) {
        double d = center - (double)j;
        double u = d * cutoff;
        double sinc;
        if (fabs(u) < 1e-9) {
          sinc = 1.0;
        } else {
          double pu = M_PI * u;
          sinc = sin(pu) / pu;
        }
        double win = 0.5 + 0.5 * cos(d * invWinDenom);
        double w = sinc * win;
        acc += (double)src[j] * w;
        wsum += w;
      }
      work[k] = wsum != 0.0 ? (acc / wsum) : 0.0;
    }
  }

  double energy = 0.0;
  for (int k = 1; k <= n; k++) {
    energy += work[k] * work[k];
  }
  double rms = sqrt(energy / (double)n);
  double intensityDb = rms > 1e-9 ? (20.0 * log10(rms)) : -120.0;

  double alpha = exp(-2.0 * M_PI * preemphHz / dstRate);
  for (int k = 0; k < n; k++) {
    dst[k] = work[k + 1] - alpha * work[k];
  }
  free(work);
  return intensityDb;
}

// Measures exact H1 and H2 harmonic amplitudes (dB) at `f0` and `2*f0` using a Hann-windowed
// 40 ms Goertzel/DFT around `centreSec`. Returns `H1 - H2` in dB (higher = lighter vocal weight).
double ov_harmonic_h1_h2(
  const float *samples,
  int sampleCount,
  double sampleRate,
  double centreSec,
  double f0
) {
  if (!samples || sampleCount <= 0 || sampleRate <= 0.0 || f0 < 40.0 || f0 * 2.2 >= sampleRate * 0.5) {
    return 0.0;
  }
  // Use ~40 ms (at least 3 full pitch periods) so harmonics are cleanly resolved.
  int winLen = (int)llround(fmax(0.040 * sampleRate, (3.2 / f0) * sampleRate));
  if (winLen < 128) winLen = 128;
  if (winLen > 4096) winLen = 4096;
  int half = winLen / 2;
  int centre = (int)llround(centreSec * sampleRate);
  int start = centre - half;

  double w1 = 2.0 * M_PI * f0 / sampleRate;
  double w2 = 2.0 * w1;
  double invN = 2.0 * M_PI / (double)winLen;

  double re1 = 0.0, im1 = 0.0;
  double re2 = 0.0, im2 = 0.0;
  for (int i = 0; i < winLen; i++) {
    int idx = start + i;
    if (idx < 0 || idx >= sampleCount) continue;
    double win = 0.5 * (1.0 - cos(invN * (double)i));
    double s = (double)samples[idx] * win;
    double a1 = w1 * (double)i;
    double a2 = w2 * (double)i;
    re1 += s * cos(a1);
    im1 -= s * sin(a1);
    re2 += s * cos(a2);
    im2 -= s * sin(a2);
  }
  double mag1 = hypot(re1, im1) + 1e-12;
  double mag2 = hypot(re2, im2) + 1e-12;
  double diffDb = 20.0 * log10(mag1 / mag2);
  if (diffDb < -20.0) diffDb = -20.0;
  if (diffDb > 30.0) diffDb = 30.0;
  return diffDb;
}

// Computes a log-spaced Long-Term Average Spectrum (LTAS) over voiced frames (in dB, normalized
// so the mean across active bins is 0 dB for easy cross-recording comparison).
// Returns 1 on success, 0 if no voiced frames.
int ov_compute_ltas(
  const float *samples,
  int sampleCount,
  double sampleRate,
  const double *voicedTimes,
  int numVoiced,
  float *outBins,
  int numBins,
  double fMin,
  double fMax
) {
  if (!samples || sampleCount <= 0 || sampleRate <= 0.0 || !voicedTimes || numVoiced <= 0 || !outBins || numBins <= 1) {
    return 0;
  }
  const int fftSize = 2048;
  const int log2n = 11;
  const int half = fftSize >> 1;
  FFTSetup setup = getFftSetup(log2n);
  if (!setup) return 0;

  float *win = (float *)malloc((size_t)fftSize * sizeof(float));
  float *frame = (float *)malloc((size_t)fftSize * sizeof(float));
  float *re = (float *)malloc((size_t)half * sizeof(float));
  float *im = (float *)malloc((size_t)half * sizeof(float));
  double *avgPow = (double *)calloc((size_t)half, sizeof(double));
  if (!win || !frame || !re || !im || !avgPow) {
    free(win); free(frame); free(re); free(im); free(avgPow);
    return 0;
  }
  for (int i = 0; i < fftSize; i++) {
    win[i] = 0.5f * (1.0f - cosf((float)(2.0 * M_PI * (double)i / (double)fftSize)));
  }

  // Cap work to at most ~240 evenly-spaced voiced frames so even long clips finish in < 3 ms.
  int stride = numVoiced > 240 ? (numVoiced / 240) : 1;
  int used = 0;
  DSPSplitComplex split = {re, im};

  for (int v = 0; v < numVoiced; v += stride) {
    int centre = (int)llround(voicedTimes[v] * sampleRate);
    int start = centre - half;
    for (int i = 0; i < fftSize; i++) {
      int idx = start + i;
      frame[i] = (idx >= 0 && idx < sampleCount) ? (samples[idx] * win[i]) : 0.0f;
    }
    vDSP_ctoz((const DSPComplex *)frame, 2, &split, 1, (vDSP_Length)half);
    vDSP_fft_zrip(setup, &split, 1, (vDSP_Length)log2n, kFFTDirection_Forward);
    avgPow[0] += (double)(re[0] * re[0]);
    for (int k = 1; k < half; k++) {
      avgPow[k] += (double)(re[k] * re[k] + im[k] * im[k]);
    }
    used++;
  }

  if (used == 0) {
    free(win); free(frame); free(re); free(im); free(avgPow);
    return 0;
  }

  double hzPerBin = sampleRate / (double)fftSize;
  double logMin = log(fmax(20.0, fMin));
  double logMax = log(fmin(sampleRate * 0.49, fMax));
  double dbMean = 0.0;

  for (int b = 0; b < numBins; b++) {
    double t0 = (double)b / (double)numBins;
    double t1 = (double)(b + 1) / (double)numBins;
    double f0 = exp(logMin + t0 * (logMax - logMin));
    double f1 = exp(logMin + t1 * (logMax - logMin));
    int k0 = (int)floor(f0 / hzPerBin);
    int k1 = (int)ceil(f1 / hzPerBin);
    if (k0 < 1) k0 = 1;
    if (k1 >= half) k1 = half - 1;
    if (k1 < k0) k1 = k0;
    double sum = 0.0;
    for (int k = k0; k <= k1; k++) {
      sum += avgPow[k];
    }
    double p = sum / ((double)(k1 - k0 + 1) * (double)used);
    double db = 10.0 * log10(p + 1e-18);
    outBins[b] = (float)db;
    dbMean += db;
  }

  dbMean /= (double)numBins;
  for (int b = 0; b < numBins; b++) {
    outBins[b] = (float)((double)outBins[b] - dbMean);
  }

  free(win); free(frame); free(re); free(im); free(avgPow);
  return 1;
}

// ============================================================================
// ============================================================================
// IDE-Style Real-Filesystem Explorer Tree (~/Documents/Overtone Library/)
// Supports nested expandable directories, renaming BOTH folders & files on disk,
// safe Trash deletion, and optional `.overtone.json` sidecar metadata per folder.
// ============================================================================

static os_unfair_lock gLibLock = OS_UNFAIR_LOCK_INIT;
static NSMutableArray<NSDictionary *> *gLibTree = nil;
static NSString *gLibFolderNotes = @"";

static NSString *libRootPath(void) {
  NSArray<NSString *> *docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  NSString *base = docs.count > 0 ? docs[0] : NSHomeDirectory();
  NSString *glintDir = [base stringByAppendingPathComponent:@"Glint Library"];
  NSString *legacyDir = [base stringByAppendingPathComponent:@"Overtone Library"];
  NSFileManager *fm = [NSFileManager defaultManager];
  if (![fm fileExistsAtPath:glintDir] && [fm fileExistsAtPath:legacyDir]) {
    return legacyDir;
  }
  return glintDir;
}

static int copyNSStringOut(NSString *s, char *out, int cap) {
  if (!out || cap <= 0) return 0;
  if (!s) {
    out[0] = 0;
    return 0;
  }
  const char *utf8 = [s UTF8String];
  if (!utf8) {
    out[0] = 0;
    return 0;
  }
  size_t len = strlen(utf8);
  if (len >= (size_t)cap) len = (size_t)cap - 1;
  memcpy(out, utf8, len);
  out[len] = 0;
  return (int)len;
}

static NSString *cleanRelDir(NSString *raw) {
  if (!raw || raw.length == 0) return @"";
  NSArray<NSString *> *parts = [raw pathComponents];
  NSMutableArray<NSString *> *safe = [NSMutableArray array];
  for (NSString *p in parts) {
    if (p.length == 0 || [p isEqualToString:@"/"] || [p isEqualToString:@"."]) continue;
    if ([p isEqualToString:@".."]) {
      if (safe.count > 0) [safe removeLastObject];
      continue;
    }
    [safe addObject:p];
  }
  return [safe componentsJoinedByString:@"/"];
}

static NSString *fullPathForRelDir(NSString *relDir) {
  NSString *clean = cleanRelDir(relDir);
  NSString *root = libRootPath();
  if (clean.length == 0) return root;
  return [root stringByAppendingPathComponent:clean];
}

static NSString *sanitizeFileComponent(NSString *raw) {
  if (!raw) return @"";
  NSString *trimmed = [raw stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
  if (trimmed.length == 0) return @"";
  NSCharacterSet *bad = [NSCharacterSet characterSetWithCharactersInString:@"/:\\*?\"<>|\n\r\t"];
  NSArray<NSString *> *parts = [trimmed componentsSeparatedByCharactersInSet:bad];
  NSString *joined = [[parts componentsJoinedByString:@"-"] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
  while ([joined hasPrefix:@"."]) {
    joined = [joined substringFromIndex:1];
  }
  if (joined.length > 96) joined = [joined substringToIndex:96];
  return joined;
}

static BOOL isAudioFilename(NSString *name) {
  NSString *ext = name.pathExtension.lowercaseString;
  return [ext isEqualToString:@"wav"] || [ext isEqualToString:@"mp3"] ||
         [ext isEqualToString:@"m4a"] || [ext isEqualToString:@"aac"] ||
         [ext isEqualToString:@"flac"] || [ext isEqualToString:@"aiff"] ||
         [ext isEqualToString:@"aif"] || [ext isEqualToString:@"caf"] ||
         [ext isEqualToString:@"ogg"];
}

static NSMutableDictionary *loadFolderSidecarLocked(NSString *dirPath) {
  NSMutableDictionary *meta = [@{
    @"version": @2,
    @"notes": @"",
    @"clips": [NSMutableDictionary dictionary],
  } mutableCopy];

  NSString *sidecarPath = [dirPath stringByAppendingPathComponent:@".overtone.json"];
  NSData *data = [NSData dataWithContentsOfFile:sidecarPath];
  if (data) {
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if ([parsed isKindOfClass:[NSDictionary class]]) {
      NSDictionary *d = (NSDictionary *)parsed;
      if ([d[@"notes"] isKindOfClass:[NSString class]]) meta[@"notes"] = d[@"notes"];
      if ([d[@"clips"] isKindOfClass:[NSDictionary class]]) {
        NSMutableDictionary *clips = [NSMutableDictionary dictionary];
        NSDictionary *rawClips = (NSDictionary *)d[@"clips"];
        for (NSString *k in rawClips) {
          if ([rawClips[k] isKindOfClass:[NSDictionary class]]) {
            clips[k] = [((NSDictionary *)rawClips[k]) mutableCopy];
          }
        }
        meta[@"clips"] = clips;
      }
      return meta;
    }
  }

  NSString *legacyPath = [dirPath stringByAppendingPathComponent:@"session.json"];
  NSData *legData = [NSData dataWithContentsOfFile:legacyPath];
  if (legData) {
    id parsed = [NSJSONSerialization JSONObjectWithData:legData options:0 error:nil];
    if ([parsed isKindOfClass:[NSDictionary class]]) {
      NSDictionary *d = (NSDictionary *)parsed;
      if ([d[@"notes"] isKindOfClass:[NSString class]]) meta[@"notes"] = d[@"notes"];
      NSMutableDictionary *clips = meta[@"clips"];
      NSArray *items = d[@"items"];
      if ([items isKindOfClass:[NSArray class]]) {
        for (id it in items) {
          if (![it isKindOfClass:[NSDictionary class]]) continue;
          NSDictionary *item = (NSDictionary *)it;
          NSString *fn = item[@"file"];
          if ([fn isKindOfClass:[NSString class]] && fn.length > 0) {
            clips[fn] = [@{
              @"rating": item[@"rating"] ?: @0,
              @"starred": item[@"starred"] ?: @NO,
              @"score": item[@"score"] ?: @(-1),
              @"comment": item[@"comment"] ?: @"",
            } mutableCopy];
          }
        }
      }
    }
  }
  return meta;
}

static void saveFolderSidecarLocked(NSString *dirPath, NSDictionary *meta) {
  [[NSFileManager defaultManager] createDirectoryAtPath:dirPath withIntermediateDirectories:YES attributes:nil error:nil];
  NSData *json = [NSJSONSerialization dataWithJSONObject:meta options:NSJSONWritingPrettyPrinted error:nil];
  if (json) {
    NSString *sidecarPath = [dirPath stringByAppendingPathComponent:@".overtone.json"];
    [json writeToFile:sidecarPath atomically:YES];
  }
}

static NSString *formatModDate(NSDate *dt) {
  if (!dt) return @"";
  NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
  fmt.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
  if ([[NSCalendar currentCalendar] isDateInToday:dt]) {
    fmt.dateFormat = @"HH:mm";
  } else {
    fmt.dateFormat = @"MMM d";
  }
  return [fmt stringFromDate:dt] ?: @"";
}

static NSString *displayNameForEntry(NSString *name, BOOL isDir) {
  if (!name) return @"";
  if (isDir) return name;
  if (isAudioFilename(name)) {
    NSString *stem = [name stringByDeletingPathExtension];
    if (stem.length > 0) return stem;
  }
  return name;
}

static void resolvedClipScores(
  NSDictionary *cm,
  NSString *activeRefPath,
  int *outOverall,
  int *outPitch,
  int *outSize,
  int *outWeight,
  int *outVowel,
  int *outVer
) {
  int ov = -1, pi = -1, sz = -1, wt = -1, vw = -1, ver = 0;
  if ([cm isKindOfClass:[NSDictionary class]] && activeRefPath.length > 0) {
    NSDictionary *byRef = [cm[@"scoresByRef"] isKindOfClass:[NSDictionary class]] ? cm[@"scoresByRef"] : nil;
    id val = byRef ? byRef[activeRefPath] : nil;
    if ([val isKindOfClass:[NSDictionary class]]) {
      NSDictionary *d = (NSDictionary *)val;
      if (d[@"overall"]) ov = [d[@"overall"] intValue];
      if (d[@"pitch"]) pi = [d[@"pitch"] intValue];
      if (d[@"size"]) sz = [d[@"size"] intValue];
      if (d[@"weight"]) wt = [d[@"weight"] intValue];
      if (d[@"vowel"]) vw = [d[@"vowel"] intValue];
      ver = d[@"v"] ? [d[@"v"] intValue] : 2;
    } else if ([val isKindOfClass:[NSNumber class]]) {
      ov = [val intValue];
      ver = 1;
    } else {
      NSString *legacyRef = [cm[@"scoreRef"] isKindOfClass:[NSString class]] ? cm[@"scoreRef"] : @"";
      if (legacyRef.length > 0 && [legacyRef isEqualToString:activeRefPath] && cm[@"score"]) {
        ov = [cm[@"score"] intValue];
        ver = 1;
      }
    }
  }
  if (outOverall) *outOverall = ov;
  if (outPitch) *outPitch = pi;
  if (outSize) *outSize = sz;
  if (outWeight) *outWeight = wt;
  if (outVowel) *outVowel = vw;
  if (outVer) *outVer = ver;
}

static int scoreForSortMode(NSDictionary *it, int sortMode) {
  if (sortMode == 2) return [it[@"scorePitch"] intValue];
  if (sortMode == 3) return [it[@"scoreSize"] intValue];
  if (sortMode == 4) return [it[@"scoreWeight"] intValue];
  if (sortMode == 5) return [it[@"scoreVowel"] intValue];
  return [it[@"score"] intValue];
}

static void scanTreeRecLocked(NSString *relDir, int depth, NSSet<NSString *> *expandedSet, NSString *activeRefPath, int sortMode) {
  if (depth > 8) return;
  NSFileManager *fm = [NSFileManager defaultManager];
  NSString *dirPath = fullPathForRelDir(relDir);
  NSArray<NSString *> *entries = [fm contentsOfDirectoryAtPath:dirPath error:nil] ?: @[];
  NSArray<NSString *> *sorted = [entries sortedArrayUsingSelector:@selector(localizedStandardCompare:)];

  NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
  NSDictionary *clipsMeta = [meta[@"clips"] isKindOfClass:[NSDictionary class]] ? meta[@"clips"] : @{};

  // Pass 1: Directories first (IDE Explorer order!)
  for (NSString *name in sorted) {
    if ([name hasPrefix:@"."]) continue;
    NSString *full = [dirPath stringByAppendingPathComponent:name];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:full isDirectory:&isDir] || !isDir) continue;
    NSString *childRel = relDir.length > 0 ? [relDir stringByAppendingPathComponent:name] : name;
    BOOL exp = [expandedSet containsObject:childRel];
    [gLibTree addObject:@{
      @"isDir": @YES,
      @"depth": @(depth),
      @"expanded": @(exp),
      @"name": name,
      @"displayName": name,
      @"relPath": childRel,
      @"parentRel": relDir,
      @"fullPath": full,
      @"modified": @"",
      @"rating": @0,
      @"starred": @NO,
      @"score": @(-1),
      @"scorePitch": @(-1),
      @"scoreSize": @(-1),
      @"scoreWeight": @(-1),
      @"scoreVowel": @(-1),
      @"scoreVer": @0,
      @"comment": @"",
    }];
    if (exp) {
      scanTreeRecLocked(childRel, depth + 1, expandedSet, activeRefPath, sortMode);
    }
  }

  // Pass 2: Audio files in this directory (optionally sorted by Overall / Pitch / Size / Weight / Vowels!)
  NSMutableArray<NSDictionary *> *fileNodes = [NSMutableArray array];
  for (NSString *name in sorted) {
    if ([name hasPrefix:@"."]) continue;
    NSString *full = [dirPath stringByAppendingPathComponent:name];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:full isDirectory:&isDir] || isDir) continue;
    if (!isAudioFilename(name)) continue;
    NSString *childRel = relDir.length > 0 ? [relDir stringByAppendingPathComponent:name] : name;
    NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
    NSDate *mod = attrs[NSFileModificationDate];
    NSDictionary *cm = [clipsMeta[name] isKindOfClass:[NSDictionary class]] ? clipsMeta[name] : @{};
    int scOv = -1, scPi = -1, scSz = -1, scWt = -1, scVw = -1, scVer = 0;
    resolvedClipScores(cm, activeRefPath, &scOv, &scPi, &scSz, &scWt, &scVw, &scVer);
    [fileNodes addObject:@{
      @"isDir": @NO,
      @"depth": @(depth),
      @"expanded": @NO,
      @"name": name,
      @"displayName": displayNameForEntry(name, NO),
      @"relPath": childRel,
      @"parentRel": relDir,
      @"fullPath": full,
      @"modified": formatModDate(mod),
      @"rating": cm[@"rating"] ?: @0,
      @"starred": cm[@"starred"] ?: @NO,
      @"score": @(scOv),
      @"scorePitch": @(scPi),
      @"scoreSize": @(scSz),
      @"scoreWeight": @(scWt),
      @"scoreVowel": @(scVw),
      @"scoreVer": @(scVer),
      @"comment": cm[@"comment"] ?: @"",
    }];
  }
  if (fileNodes.count > 1) {
    if (sortMode > 0) {
      [fileNodes sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        BOOL aIsRef = activeRefPath.length > 0 && [a[@"fullPath"] isEqualToString:activeRefPath];
        BOOL bIsRef = activeRefPath.length > 0 && [b[@"fullPath"] isEqualToString:activeRefPath];
        if (aIsRef != bIsRef) return aIsRef ? NSOrderedAscending : NSOrderedDescending;
        int sa = scoreForSortMode(a, sortMode);
        int sb = scoreForSortMode(b, sortMode);
        if (sa != sb) return sa > sb ? NSOrderedAscending : NSOrderedDescending;
        return [a[@"displayName"] localizedStandardCompare:b[@"displayName"]];
      }];
    } else {
      [fileNodes sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [a[@"displayName"] localizedStandardCompare:b[@"displayName"]];
      }];
    }
  }
  [gLibTree addObjectsFromArray:fileNodes];
}

int ov_lib_scan_tree(
  const char *expandedLines,
  size_t expandedLen,
  const char *notesDir,
  size_t notesDirLen,
  const char *activeRefPath,
  size_t activeRefLen,
  int sortMode
) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    [[NSFileManager defaultManager] createDirectoryAtPath:libRootPath() withIntermediateDirectories:YES attributes:nil error:nil];
    NSMutableSet<NSString *> *expSet = [NSMutableSet set];
    if (expandedLines && expandedLen > 0) {
      NSString *raw = [[NSString alloc] initWithBytes:expandedLines length:expandedLen encoding:NSUTF8StringEncoding];
      for (NSString *line in [raw componentsSeparatedByString:@"\n"]) {
        NSString *c = cleanRelDir(line);
        if (c.length > 0) [expSet addObject:c];
      }
    }
    NSString *refPath = (activeRefPath && activeRefLen > 0)
      ? [[NSString alloc] initWithBytes:activeRefPath length:activeRefLen encoding:NSUTF8StringEncoding]
      : @"";
    gLibTree = [NSMutableArray array];
    scanTreeRecLocked(@"", 0, expSet, refPath, sortMode);

    NSString *nRel = (notesDir && notesDirLen > 0)
      ? [[NSString alloc] initWithBytes:notesDir length:notesDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSMutableDictionary *nMeta = loadFolderSidecarLocked(fullPathForRelDir(nRel));
    gLibFolderNotes = [nMeta[@"notes"] isKindOfClass:[NSString class]] ? nMeta[@"notes"] : @"";

    int count = (int)gLibTree.count;
    os_unfair_lock_unlock(&gLibLock);
    return count;
  }
}

void ov_lib_tree_scores(
  int idx,
  int *outOverall,
  int *outPitch,
  int *outSize,
  int *outWeight,
  int *outVowel,
  int *outVer
) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    if (gLibTree && idx >= 0 && idx < (int)gLibTree.count) {
      NSDictionary *it = gLibTree[(NSUInteger)idx];
      if (outOverall) *outOverall = it[@"score"] ? [it[@"score"] intValue] : -1;
      if (outPitch) *outPitch = it[@"scorePitch"] ? [it[@"scorePitch"] intValue] : -1;
      if (outSize) *outSize = it[@"scoreSize"] ? [it[@"scoreSize"] intValue] : -1;
      if (outWeight) *outWeight = it[@"scoreWeight"] ? [it[@"scoreWeight"] intValue] : -1;
      if (outVowel) *outVowel = it[@"scoreVowel"] ? [it[@"scoreVowel"] intValue] : -1;
      if (outVer) *outVer = it[@"scoreVer"] ? [it[@"scoreVer"] intValue] : 0;
    }
    os_unfair_lock_unlock(&gLibLock);
  }
}

void ov_lib_tree_meta(
  int idx,
  int *outIsDir,
  int *outDepth,
  int *outExpanded,
  int *outRating,
  int *outStarred,
  int *outScore
) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    if (gLibTree && idx >= 0 && idx < (int)gLibTree.count) {
      NSDictionary *it = gLibTree[(NSUInteger)idx];
      if (outIsDir) *outIsDir = [it[@"isDir"] boolValue] ? 1 : 0;
      if (outDepth) *outDepth = [it[@"depth"] intValue];
      if (outExpanded) *outExpanded = [it[@"expanded"] boolValue] ? 1 : 0;
      if (outRating) *outRating = [it[@"rating"] intValue];
      if (outStarred) *outStarred = [it[@"starred"] boolValue] ? 1 : 0;
      if (outScore) *outScore = it[@"score"] ? [it[@"score"] intValue] : -1;
    }
    os_unfair_lock_unlock(&gLibLock);
  }
}

int ov_lib_tree_str(int idx, int field, char *out, int cap) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    int len = 0;
    if (gLibTree && idx >= 0 && idx < (int)gLibTree.count) {
      NSDictionary *it = gLibTree[(NSUInteger)idx];
      NSString *val = @"";
      if (field == 0) val = it[@"name"] ?: @"";
      else if (field == 1) val = it[@"relPath"] ?: @"";
      else if (field == 2) val = it[@"parentRel"] ?: @"";
      else if (field == 3) val = it[@"fullPath"] ?: @"";
      else if (field == 4) val = it[@"modified"] ?: @"";
      else if (field == 5) val = it[@"comment"] ?: @"";
      else if (field == 6) val = it[@"displayName"] ?: it[@"name"] ?: @"";
      len = copyNSStringOut(val, out, cap);
    }
    os_unfair_lock_unlock(&gLibLock);
    return len;
  }
}

int ov_lib_create_folder(
  const char *parentRel,
  size_t parentLen,
  const char *folderName,
  size_t nameLen,
  char *outRel,
  int outCap
) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    NSString *pRel = (parentRel && parentLen > 0)
      ? [[NSString alloc] initWithBytes:parentRel length:parentLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *rawName = (folderName && nameLen > 0)
      ? [[NSString alloc] initWithBytes:folderName length:nameLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *cleanName = sanitizeFileComponent(rawName);
    if (cleanName.length == 0) cleanName = @"New Folder";

    NSString *cleanParent = cleanRelDir(pRel);
    NSString *parentPath = fullPathForRelDir(cleanParent);
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:parentPath withIntermediateDirectories:YES attributes:nil error:nil];

    NSString *candidate = cleanName;
    int suffix = 2;
    while ([fm fileExistsAtPath:[parentPath stringByAppendingPathComponent:candidate]]) {
      candidate = [NSString stringWithFormat:@"%@ %d", cleanName, suffix++];
    }
    NSString *newPath = [parentPath stringByAppendingPathComponent:candidate];
    BOOL ok = [fm createDirectoryAtPath:newPath withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *newRel = cleanParent.length > 0 ? [cleanParent stringByAppendingPathComponent:candidate] : candidate;
    if (ok) {
      copyNSStringOut(newRel, outRel, outCap);
    } else if (outRel && outCap > 0) {
      outRel[0] = 0;
    }
    os_unfair_lock_unlock(&gLibLock);
    return ok ? 1 : 0;
  }
}

void ov_lib_reveal_dir(const char *relDir, size_t relDirLen) {
  @autoreleasepool {
    NSString *rel = (relDir && relDirLen > 0)
      ? [[NSString alloc] initWithBytes:relDir length:relDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *full = fullPathForRelDir(rel);
    [[NSFileManager defaultManager] createDirectoryAtPath:full withIntermediateDirectories:YES attributes:nil error:nil];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:full isDirectory:YES]];
  }
}

void ov_lib_reveal_entry(const char *parentRel, size_t parentLen, const char *name, size_t nameLen) {
  @autoreleasepool {
    NSString *pRel = (parentRel && parentLen > 0)
      ? [[NSString alloc] initWithBytes:parentRel length:parentLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(pRel);
    NSString *fn = (name && nameLen > 0)
      ? [[NSString alloc] initWithBytes:name length:nameLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *full = fn.length > 0 ? [dirPath stringByAppendingPathComponent:fn] : dirPath;
    if ([[NSFileManager defaultManager] fileExistsAtPath:full]) {
      NSURL *url = [NSURL fileURLWithPath:full];
      [[NSWorkspace sharedWorkspace] activateFileViewerSelectingURLs:@[url]];
    } else {
      [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:dirPath isDirectory:YES]];
    }
  }
}

int ov_lib_save_wav(
  const char *relDir,
  size_t relDirLen,
  const char *desiredName,
  size_t nameLen,
  int overwriteExisting,
  const float *samples,
  int count,
  int sampleRate,
  int score,
  char *outFileName,
  int outNameCap,
  char *outFullPath,
  int outPathCap
) {
  @autoreleasepool {
    if (!samples || count <= 0 || sampleRate <= 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *rel = (relDir && relDirLen > 0)
      ? [[NSString alloc] initWithBytes:relDir length:relDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(rel);
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:dirPath withIntermediateDirectories:YES attributes:nil error:nil];

    NSString *rawName = (desiredName && nameLen > 0)
      ? [[NSString alloc] initWithBytes:desiredName length:nameLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *stem = sanitizeFileComponent(rawName);
    if (isAudioFilename(stem)) {
      stem = [stem stringByDeletingPathExtension];
    }
    if (stem.length == 0) stem = @"clip";

    NSString *fileName = nil;
    if (overwriteExisting) {
      fileName = [stem stringByAppendingPathExtension:@"wav"];
    } else {
      NSString *cand = [stem stringByAppendingPathExtension:@"wav"];
      if (![fm fileExistsAtPath:[dirPath stringByAppendingPathComponent:cand]]) {
        fileName = cand;
      } else {
        int n = 2;
        while (n < 10000) {
          cand = [NSString stringWithFormat:@"%@_%02d.wav", stem, n++];
          if (![fm fileExistsAtPath:[dirPath stringByAppendingPathComponent:cand]]) {
            fileName = cand;
            break;
          }
        }
        if (!fileName) fileName = [NSString stringWithFormat:@"%@_%ld.wav", stem, (long)[[NSDate date] timeIntervalSince1970]];
      }
    }

    NSString *wavPath = [dirPath stringByAppendingPathComponent:fileName];
    const char *wavUtf8 = [wavPath UTF8String];
    int ok = ov_export_wav(wavUtf8, strlen(wavUtf8), samples, count, sampleRate);
    if (!ok) {
      os_unfair_lock_unlock(&gLibLock);
      return 0;
    }

    if (score >= 0) {
      NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
      NSMutableDictionary *clips = meta[@"clips"];
      NSMutableDictionary *entry = [clips[fileName] isKindOfClass:[NSDictionary class]]
        ? [clips[fileName] mutableCopy]
        : [@{ @"rating": @0, @"starred": @NO, @"score": @(-1), @"comment": @"" } mutableCopy];
      entry[@"score"] = @(score);
      clips[fileName] = entry;
      saveFolderSidecarLocked(dirPath, meta);
    }

    copyNSStringOut(fileName, outFileName, outNameCap);
    copyNSStringOut(wavPath, outFullPath, outPathCap);
    os_unfair_lock_unlock(&gLibLock);
    return 1;
  }
}

// Renames EITHER a real directory (isDir != 0) OR a real audio file (isDir == 0) on disk!
int ov_lib_rename_entry(
  const char *parentRel,
  size_t parentLen,
  const char *oldName,
  size_t oldLen,
  const char *newName,
  size_t newLen,
  int isDir,
  char *outNewName,
  int outNameCap,
  char *outNewRel,
  int outRelCap,
  char *outFullPath,
  int outPathCap
) {
  @autoreleasepool {
    if (!oldName || oldLen == 0 || !newName || newLen == 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *pRel = (parentRel && parentLen > 0)
      ? [[NSString alloc] initWithBytes:parentRel length:parentLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *cleanParent = cleanRelDir(pRel);
    NSString *dirPath = fullPathForRelDir(cleanParent);
    NSString *oldFn = [[NSString alloc] initWithBytes:oldName length:oldLen encoding:NSUTF8StringEncoding];
    NSString *rawNew = [[NSString alloc] initWithBytes:newName length:newLen encoding:NSUTF8StringEncoding];
    NSString *cleanNew = sanitizeFileComponent(rawNew);
    if (cleanNew.length == 0) {
      os_unfair_lock_unlock(&gLibLock);
      return 0;
    }
    if (!isDir) {
      // Lock the original audio file extension so users never edit or break it!
      NSString *oldExt = oldFn.pathExtension;
      if (oldExt.length > 0) {
        if (cleanNew.pathExtension.length > 0 &&
            [cleanNew.pathExtension caseInsensitiveCompare:oldExt] == NSOrderedSame) {
          cleanNew = [cleanNew stringByDeletingPathExtension];
        }
        cleanNew = [cleanNew stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (cleanNew.length == 0) {
          os_unfair_lock_unlock(&gLibLock);
          return 0;
        }
        cleanNew = [cleanNew stringByAppendingPathExtension:oldExt];
      }
    }
    NSString *newRel = cleanParent.length > 0 ? [cleanParent stringByAppendingPathComponent:cleanNew] : cleanNew;
    NSString *srcPath = [dirPath stringByAppendingPathComponent:oldFn];
    NSString *dstPath = [dirPath stringByAppendingPathComponent:cleanNew];
    if ([oldFn isEqualToString:cleanNew]) {
      copyNSStringOut(cleanNew, outNewName, outNameCap);
      copyNSStringOut(newRel, outNewRel, outRelCap);
      copyNSStringOut(dstPath, outFullPath, outPathCap);
      os_unfair_lock_unlock(&gLibLock);
      return 1;
    }

    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL caseOnlyRename = ([oldFn caseInsensitiveCompare:cleanNew] == NSOrderedSame);
    if (![fm fileExistsAtPath:srcPath] || (!caseOnlyRename && [fm fileExistsAtPath:dstPath])) {
      os_unfair_lock_unlock(&gLibLock);
      return 0;
    }

    NSError *err = nil;
    BOOL ok = [fm moveItemAtPath:srcPath toPath:dstPath error:&err];
    if (ok) {
      if (!isDir) {
        NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
        NSMutableDictionary *clips = meta[@"clips"];
        if (clips[oldFn]) {
          clips[cleanNew] = clips[oldFn];
          [clips removeObjectForKey:oldFn];
          saveFolderSidecarLocked(dirPath, meta);
        }
      }
      copyNSStringOut(cleanNew, outNewName, outNameCap);
      copyNSStringOut(newRel, outNewRel, outRelCap);
      copyNSStringOut(dstPath, outFullPath, outPathCap);
    }
    os_unfair_lock_unlock(&gLibLock);
    return ok ? 1 : 0;
  }
}

// Moves EITHER a real directory (isDir != 0) OR a real audio file (isDir == 0) into dstFolderRel!
int ov_lib_move_entry(
  const char *srcParentRel,
  size_t srcParentLen,
  const char *name,
  size_t nameLen,
  const char *dstFolderRel,
  size_t dstFolderLen,
  int isDir,
  char *outNewRel,
  int outRelCap,
  char *outFullPath,
  int outPathCap
) {
  @autoreleasepool {
    if (!name || nameLen == 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *srcParentRaw = (srcParentRel && srcParentLen > 0)
      ? [[NSString alloc] initWithBytes:srcParentRel length:srcParentLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dstFolderRaw = (dstFolderRel && dstFolderLen > 0)
      ? [[NSString alloc] initWithBytes:dstFolderRel length:dstFolderLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *srcParent = cleanRelDir(srcParentRaw);
    NSString *dstFolder = cleanRelDir(dstFolderRaw);
    NSString *fn = [[NSString alloc] initWithBytes:name length:nameLen encoding:NSUTF8StringEncoding];
    NSString *srcRel = srcParent.length > 0 ? [srcParent stringByAppendingPathComponent:fn] : fn;

    // Guard against no-op or moving a folder into itself / its own descendant
    if ([srcParent isEqualToString:dstFolder]) {
      os_unfair_lock_unlock(&gLibLock);
      return 0;
    }
    if (isDir) {
      if ([dstFolder isEqualToString:srcRel] ||
          [dstFolder hasPrefix:[srcRel stringByAppendingString:@"/"]]) {
        os_unfair_lock_unlock(&gLibLock);
        return 0;
      }
    }

    NSString *srcDirPath = fullPathForRelDir(srcParent);
    NSString *dstDirPath = fullPathForRelDir(dstFolder);
    NSString *srcPath = [srcDirPath stringByAppendingPathComponent:fn];
    NSString *dstPath = [dstDirPath stringByAppendingPathComponent:fn];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:dstDirPath withIntermediateDirectories:YES attributes:nil error:nil];

    if (![fm fileExistsAtPath:srcPath] || [fm fileExistsAtPath:dstPath]) {
      os_unfair_lock_unlock(&gLibLock);
      return 0;
    }

    NSError *err = nil;
    BOOL ok = [fm moveItemAtPath:srcPath toPath:dstPath error:&err];
    if (ok) {
      if (!isDir) {
        NSMutableDictionary *srcMeta = loadFolderSidecarLocked(srcDirPath);
        NSMutableDictionary *srcClips = srcMeta[@"clips"];
        if (srcClips[fn]) {
          NSMutableDictionary *dstMeta = loadFolderSidecarLocked(dstDirPath);
          NSMutableDictionary *dstClips = dstMeta[@"clips"];
          dstClips[fn] = srcClips[fn];
          [srcClips removeObjectForKey:fn];
          saveFolderSidecarLocked(srcDirPath, srcMeta);
          saveFolderSidecarLocked(dstDirPath, dstMeta);
        }
      }
      NSString *newRel = dstFolder.length > 0 ? [dstFolder stringByAppendingPathComponent:fn] : fn;
      copyNSStringOut(newRel, outNewRel, outRelCap);
      copyNSStringOut(dstPath, outFullPath, outPathCap);
    }
    os_unfair_lock_unlock(&gLibLock);
    return ok ? 1 : 0;
  }
}

int ov_lib_update_meta(
  const char *relDir,
  size_t relDirLen,
  const char *fileName,
  size_t fnLen,
  int rating,
  int starred,
  int score,
  const char *scoreRef,
  size_t scoreRefLen,
  const char *comment,
  size_t commentLen
) {
  @autoreleasepool {
    if (!fileName || fnLen == 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *rel = (relDir && relDirLen > 0)
      ? [[NSString alloc] initWithBytes:relDir length:relDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(rel);
    NSString *fn = [[NSString alloc] initWithBytes:fileName length:fnLen encoding:NSUTF8StringEncoding];
    NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
    NSMutableDictionary *clips = meta[@"clips"];
    NSMutableDictionary *entry = [clips[fn] isKindOfClass:[NSDictionary class]]
      ? [clips[fn] mutableCopy]
      : [@{ @"rating": @0, @"starred": @NO, @"score": @(-1), @"comment": @"" } mutableCopy];
    entry[@"rating"] = @(rating);
    entry[@"starred"] = @(starred != 0);
    if (score >= 0 && scoreRef && scoreRefLen > 0) {
      NSString *refKey = [[NSString alloc] initWithBytes:scoreRef length:scoreRefLen encoding:NSUTF8StringEncoding];
      NSMutableDictionary *byRef = [entry[@"scoresByRef"] isKindOfClass:[NSDictionary class]]
        ? [entry[@"scoresByRef"] mutableCopy]
        : [NSMutableDictionary dictionary];
      id existing = byRef[refKey];
      if ([existing isKindOfClass:[NSDictionary class]]) {
        NSMutableDictionary *d = [(NSDictionary *)existing mutableCopy];
        d[@"overall"] = @(score);
        byRef[refKey] = d;
      } else {
        byRef[refKey] = @(score);
      }
      entry[@"scoresByRef"] = byRef;
      entry[@"score"] = @(score);
      entry[@"scoreRef"] = refKey;
    }
    if (comment) {
      NSString *cStr = commentLen > 0
        ? [[NSString alloc] initWithBytes:comment length:commentLen encoding:NSUTF8StringEncoding]
        : @"";
      entry[@"comment"] = cStr ?: @"";
    }
    clips[fn] = entry;
    saveFolderSidecarLocked(dirPath, meta);
    os_unfair_lock_unlock(&gLibLock);
    return 1;
  }
}

int ov_lib_update_scores(
  const char *relDir,
  size_t relDirLen,
  const char *fileName,
  size_t fnLen,
  const char *scoreRef,
  size_t scoreRefLen,
  int overall,
  int pitch,
  int size,
  int weight,
  int vowel,
  int ver
) {
  @autoreleasepool {
    if (!fileName || fnLen == 0 || !scoreRef || scoreRefLen == 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *rel = (relDir && relDirLen > 0)
      ? [[NSString alloc] initWithBytes:relDir length:relDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(rel);
    NSString *fn = [[NSString alloc] initWithBytes:fileName length:fnLen encoding:NSUTF8StringEncoding];
    NSString *refKey = [[NSString alloc] initWithBytes:scoreRef length:scoreRefLen encoding:NSUTF8StringEncoding];
    NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
    NSMutableDictionary *clips = meta[@"clips"];
    NSMutableDictionary *entry = [clips[fn] isKindOfClass:[NSDictionary class]]
      ? [clips[fn] mutableCopy]
      : [@{ @"rating": @0, @"starred": @NO, @"score": @(-1), @"comment": @"" } mutableCopy];
    NSMutableDictionary *byRef = [entry[@"scoresByRef"] isKindOfClass:[NSDictionary class]]
      ? [entry[@"scoresByRef"] mutableCopy]
      : [NSMutableDictionary dictionary];
    byRef[refKey] = @{
      @"overall": @(overall),
      @"pitch": @(pitch),
      @"size": @(size),
      @"weight": @(weight),
      @"vowel": @(vowel),
      @"v": @(ver),
    };
    entry[@"scoresByRef"] = byRef;
    entry[@"score"] = @(overall);
    entry[@"scoreRef"] = refKey;
    clips[fn] = entry;
    saveFolderSidecarLocked(dirPath, meta);
    os_unfair_lock_unlock(&gLibLock);
    return 1;
  }
}

// Moves a real file OR directory safely to macOS Trash!
int ov_lib_delete_entry(const char *parentRel, size_t parentLen, const char *name, size_t nameLen, int isDir) {
  @autoreleasepool {
    if (!name || nameLen == 0) return 0;
    os_unfair_lock_lock(&gLibLock);
    NSString *rel = (parentRel && parentLen > 0)
      ? [[NSString alloc] initWithBytes:parentRel length:parentLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(rel);
    NSString *fn = [[NSString alloc] initWithBytes:name length:nameLen encoding:NSUTF8StringEncoding];
    NSString *full = [dirPath stringByAppendingPathComponent:fn];
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL removed = NO;
    if ([fm fileExistsAtPath:full]) {
      NSURL *url = [NSURL fileURLWithPath:full];
      NSError *err = nil;
      removed = [fm trashItemAtURL:url resultingItemURL:nil error:&err];
      if (!removed) {
        removed = [fm removeItemAtPath:full error:nil];
      }
    }
    if (removed && !isDir) {
      NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
      NSMutableDictionary *clips = meta[@"clips"];
      if (clips[fn]) {
        [clips removeObjectForKey:fn];
        saveFolderSidecarLocked(dirPath, meta);
      }
    }
    os_unfair_lock_unlock(&gLibLock);
    return removed ? 1 : 0;
  }
}

int ov_lib_get_notes(char *out, int cap) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    int len = copyNSStringOut(gLibFolderNotes ?: @"", out, cap);
    os_unfair_lock_unlock(&gLibLock);
    return len;
  }
}

int ov_lib_set_notes(const char *relDir, size_t relDirLen, const char *notes, size_t notesLen) {
  @autoreleasepool {
    os_unfair_lock_lock(&gLibLock);
    NSString *rel = (relDir && relDirLen > 0)
      ? [[NSString alloc] initWithBytes:relDir length:relDirLen encoding:NSUTF8StringEncoding]
      : @"";
    NSString *dirPath = fullPathForRelDir(rel);
    NSString *nStr = (notes && notesLen > 0)
      ? [[NSString alloc] initWithBytes:notes length:notesLen encoding:NSUTF8StringEncoding]
      : @"";
    gLibFolderNotes = nStr ?: @"";
    NSMutableDictionary *meta = loadFolderSidecarLocked(dirPath);
    meta[@"notes"] = gLibFolderNotes;
    saveFolderSidecarLocked(dirPath, meta);
    os_unfair_lock_unlock(&gLibLock);
    return 1;
  }
}
