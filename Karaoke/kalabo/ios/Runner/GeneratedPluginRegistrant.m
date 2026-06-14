//
//  Generated file. Do not edit.
//

// clang-format off

#import "GeneratedPluginRegistrant.h"

#if __has_include(<ffmpeg_kit_extended_flutter/FfmpegKitExtendedFlutterPlugin.h>)
#import <ffmpeg_kit_extended_flutter/FfmpegKitExtendedFlutterPlugin.h>
#else
@import ffmpeg_kit_extended_flutter;
#endif

#if __has_include(<file_picker/FilePickerPlugin.h>)
#import <file_picker/FilePickerPlugin.h>
#else
@import file_picker;
#endif

#if __has_include(<flutter_onnxruntime/FlutterOnnxruntimePlugin.h>)
#import <flutter_onnxruntime/FlutterOnnxruntimePlugin.h>
#else
@import flutter_onnxruntime;
#endif

@implementation GeneratedPluginRegistrant

+ (void)registerWithRegistry:(NSObject<FlutterPluginRegistry>*)registry {
  [FfmpegKitExtendedFlutterPlugin registerWithRegistrar:[registry registrarForPlugin:@"FfmpegKitExtendedFlutterPlugin"]];
  [FilePickerPlugin registerWithRegistrar:[registry registrarForPlugin:@"FilePickerPlugin"]];
  [FlutterOnnxruntimePlugin registerWithRegistrar:[registry registrarForPlugin:@"FlutterOnnxruntimePlugin"]];
}

@end
