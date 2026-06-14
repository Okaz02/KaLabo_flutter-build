import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'package:fftea/fftea.dart';
import 'package:ffmpeg_kit_extended_flutter/ffmpeg_kit_extended_flutter.dart';
import 'package:wav/wav.dart';

const int _kNFft = 4096;
const int _kDimF = 2048;
const int _kDimT = 256;
const int _kHopSize = 1024;

// StereoSpectrogramクラスは削除 - もう使わない

class StereoAudio {
  final Float32List left;
  final Float32List right;
  StereoAudio(this.left, this.right);
}

Future<String?> pcmAudioConverter(String inputPath) async {
  if (!await File(inputPath).exists()) {
    print('📂 ファイルが見つかりません: $inputPath');
    return null;
  }

  await Directory('C:/tmp').create(recursive: true);
  final tempInput = 'C:/tmp/uvr_input.wav';
  final tempOutput = 'C:/tmp/uvr_output.pcm';

  await File(inputPath).copy(tempInput);

  final command = '-i $tempInput -f f32le -ar 44100 -ac 2 -y $tempOutput';
  final completer = Completer<bool>();

  FFmpegKit.executeAsync(
    command,
    onComplete: (session) async {
      final returnCode = session.getReturnCode();
      completer.complete(ReturnCode.isSuccess(returnCode));
    },
  );

  final success = await completer.future;

  if (success) {
    print('✅ 変換成功: $tempOutput');
    return tempOutput;
  } else {
    print('❌ 変換失敗');
    return null;
  }
}

Future<StereoAudio> loadStereo(String path) async {
  final bytes = await File(path).readAsBytes();

  final samples = Float32List.view(
    bytes.buffer,
    bytes.offsetInBytes,
    bytes.lengthInBytes ~/ 4,
  );

  final half = samples.length ~/ 2;
  final left = Float32List(half);
  final right = Float32List(half);

  int j = 0;
  for (int i = 0; i < samples.length; i += 2) {
    left[j] = samples[i];
    right[j] = samples[i + 1];
    j++;
  }

  return StereoAudio(left, right);
}

/// STFTを実行しつつ、フレームごとに直接テンソルへ書き込む。
/// StereoSpectrogramリストを作らないのでメモリを溜め込まない。
///
/// 戻り値:
///   tensor  : shape [1, 4, _kDimF, _kDimT] のfloat32フラット配列
///   rawSpec : ISTFT用に必要な生スペクトルをフレーム×周波数で保持する
///             軽量な Float64x2List の2本だけ（left/right）
///             ※ applyMaskAndIstft() に渡す
(
  Float32List tensor,
  List<Float64x2List> specLeft,
  List<Float64x2List> specRight,
)
stftToTensor(Float32List leftSamples, Float32List rightSamples) {
  final fft = FFT(_kNFft);
  final window = Window.hanning(_kNFft);

  const int freq = _kDimF;
  const int time = _kDimT;
  final tensor = Float32List(4 * freq * time);

  // ISTFT用に生スペクトルだけ保存（フレームリスト）
  final specLeft = <Float64x2List>[];
  final specRight = <Float64x2List>[];

  final int trim = _kNFft ~/ 2;
  // 呼び出し元は _kChunkSize ぴったりのゼロパディング済みバッファを渡す前提。
  // paddedLength は常に _kChunkSize + 2*trim で固定し、末尾チャンクの範囲外アクセスを防ぐ。
  const int expectedChunkSize = _kHopSize * _kDimT; // == _kChunkSize
  final int paddedLength = expectedChunkSize + 2 * trim;
  final padLeft = Float32List(paddedLength);
  final padRight = Float32List(paddedLength);

  // leftSamples は最大 expectedChunkSize、末尾チャンクはゼロ埋め済みなのでそのままコピー
  final int copyLen = leftSamples.length < expectedChunkSize
      ? leftSamples.length
      : expectedChunkSize;
  for (int i = 0; i < copyLen; i++) {
    padLeft[trim + i] = leftSamples[i];
    padRight[trim + i] = rightSamples[i];
  }

  // ch0 = instRe_L, ch1 = instIm_L, ch2 = instRe_R, ch3 = instIm_R
  // インデックス: ch * freq * time + f * time + t
  final int ch0Base = 0 * freq * time;
  final int ch1Base = 1 * freq * time;
  final int ch2Base = 2 * freq * time;
  final int ch3Base = 3 * freq * time;

  for (int t = 0; t < time; t++) {
    final int offset = t * _kHopSize;
    final chunkL = Float32List(_kNFft);
    final chunkR = Float32List(_kNFft);

    for (int i = 0; i < _kNFft; i++) {
      chunkL[i] = padLeft[offset + i] * window[i];
      chunkR[i] = padRight[offset + i] * window[i];
    }

    final frameL = fft.realFft(chunkL).discardConjugates();
    final frameR = fft.realFft(chunkR).discardConjugates();

    // テンソルへ直接書き込み（リストに溜めない）
    for (int f = 0; f < freq && f < frameL.length; f++) {
      final vL = frameL[f];
      final vR = frameR[f];
      final int base = f * time + t;
      tensor[ch0Base + base] = vL.x.toDouble();
      tensor[ch1Base + base] = vL.y.toDouble();
      tensor[ch2Base + base] = vR.x.toDouble();
      tensor[ch3Base + base] = vR.y.toDouble();
    }

    // ISTFT用に生スペクトルは保存が必要
    specLeft.add(frameL);
    specRight.add(frameR);
  }

  return (tensor, specLeft, specRight);
}

/// マスク適用＋ISTFTを1パスで処理。
/// rawMask shape: [1, 4, _kDimF, _kDimT]
///   ch0=instRe_L, ch1=instIm_L, ch2=instRe_R, ch3=instIm_R
(
  Float32List vocalLeft,
  Float32List vocalRight,
  Float32List instLeft,
  Float32List instRight,
)
applyMaskAndIstft({
  required Float32List rawMask,
  required int frameSize,
  required List<Float64x2List> specLeft,
  required List<Float64x2List> specRight,
}) {
  final int time = specLeft.length;
  final int paddedLength = (time - 1) * _kHopSize + _kNFft;

  final outVL = Float64List(paddedLength);
  final outVR = Float64List(paddedLength);
  final outIL = Float64List(paddedLength);
  final outIR = Float64List(paddedLength);
  final windowSum = Float64List(paddedLength);

  final window = Window.hanning(_kNFft);
  final fft = FFT(_kNFft);

  final int halfFft = _kNFft ~/ 2 + 1;

  for (int t = 0; t < time; t++) {
    final frameL = specLeft[t];
    final frameR = specRight[t];

    // マスク適用してボーカル/インスト分離スペクトルを作る
    final vSpecL = Float64x2List(halfFft);
    final vSpecR = Float64x2List(halfFft);
    final iSpecL = Float64x2List(halfFft);
    final iSpecR = Float64x2List(halfFft);

    for (int f = 0; f < _kDimF && f < frameL.length; f++) {
      final int base = f * _kDimT + t;
      final double mRe0 = rawMask[0 * frameSize + base];
      final double mIm0 = rawMask[1 * frameSize + base];
      final double mRe1 = rawMask[2 * frameSize + base];
      final double mIm1 = rawMask[3 * frameSize + base];

      final inL = frameL[f];
      final inR = frameR[f];

      vSpecL[f] = Float64x2(inL.x - mRe0, inL.y - mIm0);
      vSpecR[f] = Float64x2(inR.x - mRe1, inR.y - mIm1);
      iSpecL[f] = Float64x2(mRe0, mIm0);
      iSpecR[f] = Float64x2(mRe1, mIm1);
    }

    // ISTFT（ボーカル）
    _istftFrame(
      fft,
      window,
      vSpecL,
      outVL,
      windowSum,
      t,
      accumWindowSum: false,
    );
    _istftFrame(
      fft,
      window,
      vSpecR,
      outVR,
      windowSum,
      t,
      accumWindowSum: false,
    );
    // ISTFT（インスト）＋windowSumはこちらで積算
    _istftFrame(
      fft,
      window,
      iSpecL,
      outIL,
      windowSum,
      t,
      accumWindowSum: false,
    );
    _istftFrame(fft, window, iSpecR, outIR, windowSum, t, accumWindowSum: true);
  }

  // overlap-add の正規化
  for (int i = 0; i < paddedLength; i++) {
    if (windowSum[i] > 1e-8) {
      outVL[i] /= windowSum[i];
      outVR[i] /= windowSum[i];
      outIL[i] /= windowSum[i];
      outIR[i] /= windowSum[i];
    }
  }

  // トリム
  final int trim = _kNFft ~/ 2;
  final int targetLength = paddedLength - 2 * trim;

  final finalVL = Float32List(targetLength);
  final finalVR = Float32List(targetLength);
  final finalIL = Float32List(targetLength);
  final finalIR = Float32List(targetLength);

  for (int i = 0; i < targetLength; i++) {
    finalVL[i] = outVL[trim + i].toDouble();
    finalVR[i] = outVR[trim + i].toDouble();
    finalIL[i] = outIL[trim + i].toDouble();
    finalIR[i] = outIR[trim + i].toDouble();
  }

  return (finalVL, finalVR, finalIL, finalIR);
}

/// 1フレーム分のISTFTを overlap-add バッファに加算する。
/// [accumWindowSum] が true のときだけ windowSum を更新する（4チャンネルで1回だけ積算したい）。
void _istftFrame(
  FFT fft,
  Float64List window,
  Float64x2List halfSpec,
  Float64List outBuf,
  Float64List windowSum,
  int t, {
  required bool accumWindowSum,
}) {
  final int offset = t * _kHopSize;
  final fullSpec = Float64x2List(_kNFft);
  final int half = _kNFft ~/ 2 + 1;

  for (int f = 0; f < half && f < halfSpec.length; f++) {
    fullSpec[f] = halfSpec[f];
  }
  for (int f = half; f < _kNFft; f++) {
    final int mirror = _kNFft - f;
    if (mirror < halfSpec.length) {
      fullSpec[f] = Float64x2(halfSpec[mirror].x, -halfSpec[mirror].y);
    }
  }

  final timeFrame = fft.realInverseFft(fullSpec);

  for (int i = 0; i < _kNFft; i++) {
    final double w = window[i];
    outBuf[offset + i] += (timeFrame[i] * w).toDouble();
    if (accumWindowSum) {
      windowSum[offset + i] += w * w;
    }
  }
}

Future<String> saveWav(
  Float32List left,
  Float32List right,
  String outputPath,
  int sampleRate,
) async {
  final wav = Wav(
    [Float64List.fromList(left), Float64List.fromList(right)],
    sampleRate,
    WavFormat.pcm16bit,
  );
  await wav.writeFile(outputPath);
  print('✅ WAV保存完了: $outputPath');
  return outputPath;
}
