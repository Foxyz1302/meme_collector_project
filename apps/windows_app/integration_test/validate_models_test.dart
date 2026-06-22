/// ONNX + model2vec validation script.
///
/// Run from apps/windows_app/:
///   flutter test integration_test/validate_models_test.dart -d windows
///
/// This is the GATE before writing any real inference code. If any model
/// fails to load or produces garbage output, fix or find an alternative
/// before proceeding.
///
/// Uses onnxruntime_v2 with appendDefaultProviders() which auto-selects the
/// best available execution provider:
///   - If CUDA Toolkit + cuDNN are installed → CUDA EP (fastest on NVIDIA)
///   - If TensorRT is installed → TensorRT EP (fastest overall)
///   - If full DX12 GPU support → DirectML EP
///   - Otherwise → CPU EP (always works)
///
/// On the dev machine (GTX 1080, no full DX12, CUDA toolkit uninstalled),
/// this currently falls back to CPU. Performance is still acceptable:
///   CLIP text:  ~71ms per query (well within 150ms debounce)
///   CLIP vision: ~77ms per image (fine for background ingest)
///   PP-OCR det:  ~1.3s on 960px (will be ~90ms on 256px thumbnails)
///   PP-OCR rec:  ~13ms per text region
///
/// Future GPU path: install CUDA Toolkit + cuDNN on Windows, or move to the
/// Linux server with a 3080 Ti where CUDA EP will auto-load.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:path/path.dart' as p;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Models live at workspace_root/assets/models/. From apps/windows_app/,
  // that's ../../assets/models/.
  final modelsDir = p.normalize(
      p.join(Directory.current.path, '..', '..', 'assets', 'models'));
  final report = <String, ModelReport>{};

  print('Models dir: $modelsDir');

  setUpAll(() {
    OrtEnv.instance.init();
  });

  test('detect available execution providers', () async {
    final providers = OrtEnv.instance.availableProviders();
    final providerNames = providers.map((p) => p.toString()).toList();
    print('\n>>> Available ORT execution providers: $providerNames');
    report['execution_providers'] = ModelReport(
      loaded: true,
      sanityPassed: true,
      inputNames: providerNames,
    );
  });

  test('validate CLIP text tower (text_model_fp16.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'clip_text',
      modelPath: p.join(modelsDir, 'clip_text_fp16.onnx'),
      syntheticInputs: _syntheticClipTextInputs(),
      expectedOutputLen: 512,
    );
    report['clip_text'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP text failed to load');
    expect(result.sanityPassed, true,
        reason: 'CLIP text output length != 512');
  });

  test('validate CLIP vision tower (vision_model_fp16.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'clip_vision',
      modelPath: p.join(modelsDir, 'clip_vision_fp16.onnx'),
      syntheticInputs: _syntheticClipVisionInputs(),
      expectedOutputLen: 512,
    );
    report['clip_vision'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP vision failed to load');
    expect(result.sanityPassed, true,
        reason: 'CLIP vision output length != 512');
  });

  test('validate PP-OCRv5 detection (det.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'ppocr_det',
      modelPath: p.join(modelsDir, 'ppocr_det.onnx'),
      syntheticInputs: _syntheticPpocrDetInputs(),
      expectedOutputLen: null, // variable: 960*960 = 921600 for this input size
    );
    report['ppocr_det'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'PP-OCR det failed to load');
  });

  test('validate PP-OCRv5 recognition (rec.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'ppocr_rec',
      modelPath: p.join(modelsDir, 'ppocr_rec.onnx'),
      syntheticInputs: _syntheticPpocrRecInputs(),
      expectedOutputLen: null, // 40 * 438 = 17520
    );
    report['ppocr_rec'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'PP-OCR rec failed to load');
  });

  test('validate model2vec (potion-base-32M) file presence', () async {
    final result = _validateModel2VecFiles(p.join(modelsDir, 'potion-base-32M'));
    report['model2vec'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'model2vec files missing');
  });

  tearDownAll(() async {
    final reportPath = p.normalize(p.join(
        Directory.current.path, '..', '..', 'scripts', 'validation_report.json'));
    final reportFile = File(reportPath);
    await reportFile.parent.create(recursive: true);
    final reportJson = report.map((k, v) => MapEntry(k, v.toJson()));
    await reportFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(reportJson),
      flush: true,
    );

    print('\n${'=' * 60}');
    print('VALIDATION REPORT');
    print('${'=' * 60}');
    for (final entry in report.entries) {
      final r = entry.value;
      print('\n${entry.key}:');
      print('  loaded:        ${r.loaded}');
      print('  load_ms:       ${r.loadMs?.toStringAsFixed(1) ?? 'N/A'}');
      print('  infer_ms:      ${r.inferMs?.toStringAsFixed(1) ?? 'N/A'}');
      print('  output_len:    ${r.outputLen ?? 'N/A'}');
      print('  sanity_passed: ${r.sanityPassed}');
      if (r.error != null) print('  error:         ${r.error}');
      if (r.inputNames != null) print('  input_names:   ${r.inputNames}');
      if (r.outputNames != null) print('  output_names:  ${r.outputNames}');
    }
    print('\nReport saved to: $reportPath');
  });
}

// ─── Report data class ──────────────────────────────────────────────────────

class ModelReport {
  final bool loaded;
  final double? loadMs;
  final double? inferMs;
  final int? outputLen;
  final List<String>? inputNames;
  final List<String>? outputNames;
  final bool sanityPassed;
  final String? error;

  ModelReport({
    required this.loaded,
    this.loadMs,
    this.inferMs,
    this.outputLen,
    this.inputNames,
    this.outputNames,
    required this.sanityPassed,
    this.error,
  });

  Map<String, dynamic> toJson() => {
        'loaded': loaded,
        if (loadMs != null) 'load_ms': loadMs,
        if (inferMs != null) 'infer_ms': inferMs,
        if (outputLen != null) 'output_len': outputLen,
        if (inputNames != null) 'input_names': inputNames,
        if (outputNames != null) 'output_names': outputNames,
        'sanity_passed': sanityPassed,
        if (error != null) 'error': error,
      };
}

// ─── ONNX model validation ──────────────────────────────────────────────────

class _SyntheticInput {
  final String name;
  final List<int> shape;
  final List<dynamic> data;

  _SyntheticInput({
    required this.name,
    required this.shape,
    required this.data,
  });
}

Future<ModelReport> _validateOnnxModel({
  required String name,
  required String modelPath,
  required List<_SyntheticInput> syntheticInputs,
  int? expectedOutputLen,
}) async {
  final sw = Stopwatch()..start();
  try {
    final modelFile = File(modelPath);
    if (!await modelFile.exists()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'Model file not found: $modelPath',
      );
    }

    final modelBytes = await modelFile.readAsBytes();

    // Auto-select best available EP (CUDA > TensorRT > DirectML > CPU)
    final sessionOptions = OrtSessionOptions();
    sessionOptions.appendDefaultProviders();

    final session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    sw.stop();
    final loadMs = sw.elapsedMilliseconds.toDouble();

    final inputNames = session.inputNames;
    final outputNames = session.outputNames;

    print('  [$name] inputs:  $inputNames');
    print('  [$name] outputs: $outputNames');

    sw..reset()..start();

    final inputsMap = <String, OrtValue>{};
    final inputsToRelease = <OrtValue>[];
    for (final synth in syntheticInputs) {
      final tensor =
          OrtValueTensor.createTensorWithDataList(synth.data, synth.shape);
      inputsMap[synth.name] = tensor;
      inputsToRelease.add(tensor);
    }

    final runOptions = OrtRunOptions();
    final outputs = await session.runAsync(runOptions, inputsMap);

    sw.stop();
    final inferMs = sw.elapsedMilliseconds.toDouble();

    for (final t in inputsToRelease) {
      t.release();
    }
    runOptions.release();

    int? actualOutputLen;
    if (outputs != null && outputs.isNotEmpty) {
      final firstOutput = outputs.first;
      if (firstOutput != null) {
        try {
          final value = firstOutput.value;
          final flattened = _flattenToList(value);
          actualOutputLen = flattened.length;
          print('  [$name] output len: ${flattened.length}');
        } catch (e) {
          print('  [$name] could not read output: $e');
        }
        firstOutput.release();
      }
      for (var i = 1; i < outputs.length; i++) {
        outputs[i]?.release();
      }
    }

    session.release();

    var shapeOk = true;
    if (expectedOutputLen != null && actualOutputLen != null) {
      shapeOk = actualOutputLen == expectedOutputLen;
    }

    return ModelReport(
      loaded: true,
      loadMs: loadMs,
      inferMs: inferMs,
      outputLen: actualOutputLen,
      inputNames: inputNames,
      outputNames: outputNames,
      sanityPassed: shapeOk,
      error: shapeOk
          ? null
          : 'Output len $actualOutputLen != expected $expectedOutputLen',
    );
  } catch (e, st) {
    sw.stop();
    print('  [$name] EXCEPTION: $e');
    print('  [$name] STACK: $st');
    return ModelReport(
      loaded: false,
      sanityPassed: false,
      error: e.toString(),
    );
  }
}

// ─── Synthetic input generators ─────────────────────────────────────────────

/// CLIP text tower expects only one input: input_ids (int32[1, 77]).
List<_SyntheticInput> _syntheticClipTextInputs() {
  final seqLen = 77;
  final inputIds = List<int>.filled(seqLen, 0);
  inputIds[0] = 49406; // BOS
  inputIds[1] = 320; // "a"
  inputIds[2] = 2368; // "cat"
  inputIds[seqLen - 1] = 49407; // EOS
  return [
    _SyntheticInput(name: 'input_ids', shape: [1, seqLen], data: inputIds),
  ];
}

/// CLIP vision tower expects: pixel_values (float32[1, 3, 224, 224]).
List<_SyntheticInput> _syntheticClipVisionInputs() {
  final width = 224, height = 224, channels = 3;
  final total = channels * height * width;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = (rng.nextDouble() * 2 - 1).toDouble();
  }
  return [
    _SyntheticInput(
        name: 'pixel_values',
        shape: [1, channels, height, width],
        data: pixelValues),
  ];
}

/// PP-OCRv5 detection model input: x (float32[1, 3, 960, 960]).
List<_SyntheticInput> _syntheticPpocrDetInputs() {
  final size = 960;
  final total = 3 * size * size;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = rng.nextDouble().toDouble();
  }
  return [
    _SyntheticInput(name: 'x', shape: [1, 3, size, size], data: pixelValues),
  ];
}

/// PP-OCRv5 recognition model input: x (float32[1, 3, 48, 320]).
List<_SyntheticInput> _syntheticPpocrRecInputs() {
  final h = 48, w = 320;
  final total = 3 * h * w;
  final rng = math.Random(42);
  final pixelValues = Float32List(total);
  for (var i = 0; i < total; i++) {
    pixelValues[i] = rng.nextDouble().toDouble();
  }
  return [
    _SyntheticInput(name: 'x', shape: [1, 3, h, w], data: pixelValues),
  ];
}

// ─── model2vec file presence check ──────────────────────────────────────────

ModelReport _validateModel2VecFiles(String modelDir) {
  try {
    final dir = Directory(modelDir);
    if (!dir.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'Model dir not found: $modelDir',
      );
    }

    final modelFile = File(p.join(modelDir, 'model.safetensors'));
    final tokenizerFile = File(p.join(modelDir, 'tokenizer.json'));
    if (!modelFile.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'model.safetensors not found',
      );
    }
    if (!tokenizerFile.existsSync()) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'tokenizer.json not found',
      );
    }

    final modelSize = modelFile.lengthSync();
    if (modelSize < 50 * 1024 * 1024) {
      return ModelReport(
        loaded: false,
        sanityPassed: false,
        error: 'model.safetensors too small: $modelSize bytes',
      );
    }

    return ModelReport(
      loaded: true,
      sanityPassed: true,
      inputNames: const ['(files only — real API wiring in next step)'],
    );
  } catch (e) {
    return ModelReport(
      loaded: false,
      sanityPassed: false,
      error: e.toString(),
    );
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Recursively flatten a nested List structure into a flat List.
List<dynamic> _flattenToList(dynamic value) {
  final result = <dynamic>[];
  void recurse(dynamic v) {
    if (v is List) {
      for (final item in v) {
        recurse(item);
      }
    } else {
      result.add(v);
    }
  }

  recurse(value);
  return result;
}
