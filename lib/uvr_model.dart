import 'dart:io' show Platform, ProcessInfo;
import 'dart:math' show cos, pi;
import 'dart:typed_data';
import 'package:kalabo/audio_processor.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

const int _kSampleRate = 44100;
const int _kNFft = 4096;
const int _kHopLength = 1024;
const int _kDimF = 2048;
const int _kDimT = 256;
const double _kOverlap = 0.5;
const double _kNormTarget = 0.90;

const int _kTrim = _kNFft ~/ 2;
const int _kChunkSize = _kHopLength * _kDimT;
const int _kGenSize = _kChunkSize - 2 * _kTrim;

const String _kModelPathFp16 = 'assets/models/UVR_MDXNET_3_9662_fp16.onnx';
const String _kModelPathFp32 = 'assets/models/UVR_MDXNET_3_9662.onnx';

double _mb(int bytes) => (bytes / 1024 / 1024 * 10).round() / 10;

class PreparedAudio {
  final Float32List left;
  final Float32List right;
  final int originalLength;
  final int paddedLength;
  final double peak;

  PreparedAudio({
    required this.left,
    required this.right,
    required this.originalLength,
    required this.paddedLength,
    required this.peak,
  });
}

class OnnxOutputs {
  final double outPeak;
  final Float32List vocalLeft;
  final Float32List vocalRight;
  final Float32List instLeft;
  final Float32List instRight;

  OnnxOutputs({
    required this.outPeak,
    required this.vocalLeft,
    required this.vocalRight,
    required this.instLeft,
    required this.instRight,
  });
}

Future<(String, String)?> runUvr(
  String inputPath, [
  String tmpVocalPath = 'C:/tmp/uvr_vocal.wav',
  String tmpInstPath = 'C:/tmp/uvr_inst.wav',
]) async {
  print("1. FFmpeg変換を開始します...");
  final String? rawAudioPath = await pcmAudioConverter(inputPath);
  if (rawAudioPath == null) return null;

  print("2. ステレオ波形を読み込みます...");
  final audio = await preparAudio(rawAudioPath);

  print("3. チャンク推論を開始します...");
  final onnxOutput = await runOnnxModel(
    audio.left,
    audio.right,
    audio.originalLength,
    audio.paddedLength,
  );

  print("4. スケーリング処理...");
  double targetPeak = audio.peak > 0 ? audio.peak : 0.95;
  if (targetPeak > 0.95) targetPeak = 0.95;

  final double finalScale = (onnxOutput.outPeak > 0)
      ? (targetPeak / onnxOutput.outPeak)
      : 1.0;

  for (int i = 0; i < audio.originalLength; i++) {
    onnxOutput.vocalLeft[i] *= finalScale;
    onnxOutput.vocalRight[i] *= finalScale;
    onnxOutput.instLeft[i] *= finalScale;
    onnxOutput.instRight[i] *= finalScale;
  }

  print("5. WAVファイルを保存します...");
  await saveWav(
    onnxOutput.vocalLeft,
    onnxOutput.vocalRight,
    tmpVocalPath,
    _kSampleRate,
  );
  await saveWav(
    onnxOutput.instLeft,
    onnxOutput.instRight,
    tmpInstPath,
    _kSampleRate,
  );

  print("✅ 全工程が完了しました: $tmpVocalPath");
  return (tmpInstPath, tmpVocalPath);
}

Float32List _buildHanningWindow(int size) {
  final window = Float32List(size);
  for (int n = 0; n < size; n++) {
    window[n] = 0.5 * (1.0 - cos(2.0 * pi * n / (size - 1)));
  }
  return window;
}

Future<PreparedAudio> preparAudio(String rawAudioPath) async {
  final StereoAudio rawAudio = await loadStereo(rawAudioPath);
  final int originalLength = rawAudio.left.length;

  double peak = 0.0;
  for (int i = 0; i < originalLength; i++) {
    final double al = rawAudio.left[i].abs();
    final double ar = rawAudio.right[i].abs();
    if (al > peak) peak = al;
    if (ar > peak) peak = ar;
  }

  final double normScale = (peak > 0) ? (_kNormTarget / peak) : 1.0;
  final normLeft = Float32List(originalLength);
  final normRight = Float32List(originalLength);
  for (int i = 0; i < originalLength; i++) {
    normLeft[i] = rawAudio.left[i] * normScale;
    normRight[i] = rawAudio.right[i] * normScale;
  }

  final int remainder = originalLength % _kGenSize;
  final int pad = (remainder == 0) ? 0 : (_kGenSize + _kTrim - remainder);
  final int paddedLength = 2 * _kTrim + originalLength + pad;

  final paddedLeft = Float32List(paddedLength);
  final paddedRight = Float32List(paddedLength);
  for (int i = 0; i < originalLength; i++) {
    paddedLeft[_kTrim + i] = normLeft[i];
    paddedRight[_kTrim + i] = normRight[i];
  }

  return PreparedAudio(
    left: paddedLeft,
    right: paddedRight,
    originalLength: originalLength,
    paddedLength: paddedLength,
    peak: peak,
  );
}

Future<OnnxOutputs> runOnnxModel(
  Float32List paddedLeft,
  Float32List paddedRight,
  int originalLength,
  int paddedLength,
) async {
  final ort = OnnxRuntime();
  final String modelPath = Platform.isWindows
      ? _kModelPathFp32
      : _kModelPathFp16;
  final session = await ort.createSessionFromAsset(modelPath);

  final resultVocalLeft = Float32List(paddedLength);
  final resultVocalRight = Float32List(paddedLength);
  final resultInstLeft = Float32List(paddedLength);
  final resultInstRight = Float32List(paddedLength);
  final divider = Float32List(paddedLength);

  final int stride = ((1.0 - _kOverlap) * _kChunkSize).floor();
  int startIdx = 0;
  int chunkIndex = 0;

  while (startIdx < paddedLength) {
    final int actualLen = ((startIdx + _kChunkSize) > paddedLength)
        ? (paddedLength - startIdx)
        : _kChunkSize;

    final chunkLeft = Float32List(_kChunkSize); // ゼロ初期化済み
    final chunkRight = Float32List(_kChunkSize);
    for (int i = 0; i < actualLen; i++) {
      chunkLeft[i] = paddedLeft[startIdx + i];
      chunkRight[i] = paddedRight[startIdx + i];
    }

    final memBeforeStft = ProcessInfo.currentRss;

    final (tensor, specLeft, specRight) = stftToTensor(chunkLeft, chunkRight);

    final memAfterStft = ProcessInfo.currentRss;

    final ortInputF32 = await OrtValue.fromList(tensor, [1, 4, _kDimF, _kDimT]);
    final OrtValue ortInput;
    if (Platform.isWindows) {
      ortInput = ortInputF32;
    } else {
      ortInput = await ortInputF32.to(OrtDataType.float16);
      await ortInputF32.dispose();
    }

    final outputs = await session.run({'input': ortInput});

    final memAfterInference = ProcessInfo.currentRss;

    print(
      '[chunk $chunkIndex] '
      'STFT前: ${_mb(memBeforeStft)} MB  '
      'STFT後: ${_mb(memAfterStft)} MB (+${_mb(memAfterStft - memBeforeStft)} MB)  '
      '推論後: ${_mb(memAfterInference)} MB (+${_mb(memAfterInference - memAfterStft)} MB)',
    );

    final rawOutDynamic = await outputs['output']!.asFlattenedList();
    await ortInput.dispose();
    await outputs['output']!.dispose();
    final rawOut = Float32List(rawOutDynamic.length);
    for (int i = 0; i < rawOutDynamic.length; i++) {
      rawOut[i] = (rawOutDynamic[i] as num).toDouble();
    }

    final int frameSize = _kDimF * _kDimT;

    final (vocalLeft, vocalRight, instLeft, instRight) = applyMaskAndIstft(
      rawMask: rawOut,
      frameSize: frameSize,
      specLeft: specLeft,
      specRight: specRight,
    );

    for (int i = 0; i < actualLen; i++) {
      resultVocalLeft[startIdx + i] += i < vocalLeft.length
          ? vocalLeft[i]
          : 0.0;
      resultVocalRight[startIdx + i] += i < vocalRight.length
          ? vocalRight[i]
          : 0.0;
      resultInstLeft[startIdx + i] += i < instLeft.length ? instLeft[i] : 0.0;
      resultInstRight[startIdx + i] += i < instRight.length
          ? instRight[i]
          : 0.0;
      divider[startIdx + i] += 1.0;
    }

    final memAfterDispose = ProcessInfo.currentRss;
    print(
      '[chunk $chunkIndex] dispose後: ${_mb(memAfterDispose)} MB '
      '(解放: ${_mb(memAfterInference - memAfterDispose)} MB)',
    );

    chunkIndex++;
    startIdx += stride;
  }

  await session.close();

  final outputVocalLeft = Float32List(originalLength);
  final outputVocalRight = Float32List(originalLength);
  final outputInstLeft = Float32List(originalLength);
  final outputInstRight = Float32List(originalLength);
  double outPeak = 0.0;

  for (int i = 0; i < originalLength; i++) {
    final int idx = _kTrim + i;
    final double d = divider[idx];

    final double vL = d > 1e-8 ? resultVocalLeft[idx] / d : 0.0;
    final double vR = d > 1e-8 ? resultVocalRight[idx] / d : 0.0;
    final double iL = d > 1e-8 ? resultInstLeft[idx] / d : 0.0;
    final double iR = d > 1e-8 ? resultInstRight[idx] / d : 0.0;

    outputVocalLeft[i] = vL;
    outputVocalRight[i] = vR;
    outputInstLeft[i] = iL;
    outputInstRight[i] = iR;

    if (iL.abs() > outPeak) outPeak = iL.abs();
    if (iR.abs() > outPeak) outPeak = iR.abs();
  }

  return OnnxOutputs(
    outPeak: outPeak,
    vocalLeft: outputVocalLeft,
    vocalRight: outputVocalRight,
    instLeft: outputInstLeft,
    instRight: outputInstRight,
  );
}
