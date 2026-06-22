/// ONNX + model2vec validation script.
///
/// Run from apps/windows_app/:
///   flutter test integration_test/validate_models_test.dart -d windows
///
/// This is the GATE before writing any real inference code. If any model
/// fails to load or produces garbage output, fix or find an alternative
/// before proceeding.
///
/// What it validates:
///   1. CLIP text tower (text_model_int8.onnx) — loads, accepts synthetic input, output shape
///   2. CLIP vision tower (vision_model_int8.onnx) — loads, accepts synthetic input, output shape
///   3. PP-OCRv5 detection (det.onnx) — loads, accepts synthetic input
///   4. PP-OCRv5 recognition (rec.onnx) — loads, accepts synthetic input
///   5. model2vec (potion-base-32M) — file presence check (real API wiring TODO)
///
/// What it does NOT validate (yet):
///   - CLIP BPE tokenization (hardcoded token IDs used for now)
///   - CLIP image preprocessing (resize, normalize, transpose)
///   - PP-OCRv5 full pipeline (detection → crop → recognition)
///   - Semantic quality of CLIP embeddings (would need real tokenization)
///
/// These will be validated when the real inference implementations are written.
/// For now, we just verify the ONNX files are valid and the runtime works.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Models live at workspace_root/assets/models/. From apps/windows_app/,
  // that's ../../assets/models/.
  final modelsDir =
      p.normalize(p.join(Directory.current.path, '..', '..', 'assets', 'models'));
  final report = <String, ModelReport>{};

  print('Models dir: $modelsDir');

  test('validate CLIP text tower (text_model_int8.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'clip_text',
      modelPath: p.join(modelsDir, 'clip_text_int8.onnx'),
      syntheticInputs: _syntheticClipTextInputs(),
      outputShape: [1, 512],
    );
    report['clip_text'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP text failed to load');
  });

  test('validate CLIP vision tower (vision_model_q4.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'clip_vision',
      modelPath: p.join(modelsDir, 'clip_vision_q4.onnx'),
      syntheticInputs: _syntheticClipVisionInputs(),
      outputShape: [1, 512],
    );
    report['clip_vision'] = result;
    expect(result.loaded, true,
        reason: result.error ?? 'CLIP vision failed to load');
  });

  test('validate PP-OCRv5 detection (det.onnx)', () async {
    final result = await _validateOnnxModel(
      name: 'ppocr_det',
      modelPath: p.join(modelsDir, 'ppocr_det.onnx'),
      syntheticInputs: _syntheticPpocrDetInputs(),
      outputShape: null, // detection output shape varies, just check it runs
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
      outputShape: null,
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
    // Write the report to disk for inspection
    final reportPath = p.normalize(
        p.join(Directory.current.path, '..', '..', 'scripts', 'validation_report.json'));
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
      print('  output_shape:  ${r.outputShape ?? 'N/A'}');
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
  final List<int>? outputShape;
  final List<String>? inputNames;
  final List<String>? outputNames;
  final bool sanityPassed;
  final String? error;

  ModelReport({
    required this.loaded,
    this.loadMs,
    this.inferMs,
    this.outputShape,
    this.inputNames,
    this.outputNames,
    required this.sanityPassed,
    this.error,
  });

  Map<String, dynamic> toJson() => {
        'loaded': loaded,
        if (loadMs != null) 'load_ms': loadMs,
        if (inferMs != null) 'infer_ms': inferMs,
        if (outputShape != null) 'output_shape': outputShape,
        if (inputNames != null) 'input_names': inputNames,
        if (outputNames != null) 'output_names': outputNames,
        'sanity_passed': sanityPassed,
        if (error != null) 'error': error,
      };
}

// ─── ONNX model validation ──────────────────────────────────────────────────

/// Inputs to feed to the ONNX model. The input names + shapes are auto-
/// discovered from the session metadata.
class _SyntheticInput {
  final String name;
  final List<int> shape;
  final List<dynamic> data; // List<double> or List<int>

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
  List<int>? outputShape,
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

    final ort = OnnxRuntime();
    final session = await ort.createSession(modelPath);

    sw.stop();
    final loadMs = sw.elapsedMilliseconds.toDouble();

    // Get input/output metadata from the session.
    // inputNames and outputNames are sync properties (not async getters).
    final inputNames = session.inputNames;
    final outputNames = session.outputNames;

    print('  [$name] inputs:  $inputNames');
    print('  [$name] outputs: $outputNames');

    // Try inference with synthetic input
    sw..reset()..start();

    final inputsMap = <String, OrtValue>{};
    for (final synth in syntheticInputs) {
      // OrtValue.fromList takes (dynamic data, List<int> shape).
      // The data parameter should be a List (List<int> or List<double>).
      final tensor = await OrtValue.fromList(synth.data, synth.shape);
      inputsMap[synth.name] = tensor;
    }

    final outputs = await session.run(inputsMap);
    sw.stop();
    final inferMs = sw.elapsedMilliseconds.toDouble();

    // Dispose input tensors
    for (final t in inputsMap.values) {
      await t.dispose();
    }

    // Get output shape
    List<int>? actualShape;
    if (outputs.isNotEmpty) {
      final firstKey = outputs.keys.first;
      final firstOutput = outputs[firstKey]!;
      actualShape = firstOutput.shape;
      // Try to read the output values
      try {
        final outputList = await firstOutput.asList();
        print('  [$name] output len: ${outputList.length}');
        if (outputList.isNotEmpty) {
          print('  [$name] output[0..5]: ${outputList.take(5).toList()}');
        }
      } catch (e) {
        print('  [$name] could not read output: $e');
      }
      await firstOutput.dispose();
    }

    await session.close();

    // Verify output shape if expected
    var shapeOk = true;
    if (outputShape != null && actualShape != null) {
      // Check that the last dimension matches (batch may differ)
      if (actualShape.length >= 2 && outputShape.length >= 2) {
        shapeOk = actualShape.last == outputShape.last;
      } else {
        shapeOk = _shapesMatch(actualShape, outputShape);
      }
    }

    return ModelReport(
      loaded: true,
      loadMs: loadMs,
      inferMs: inferMs,
      outputShape: actualShape,
      inputNames: inputNames,
      outputNames: outputNames,
      sanityPassed: shapeOk,
      error: shapeOk ? null : 'Output shape $actualShape != expected $outputShape',
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

/// CLIP text tower expects only one input:
///   - input_ids: int32[1, 77]
///
/// Note: it does NOT accept attention_mask (the text tower uses causal
/// attention internally — no mask needed). Passing attention_mask causes:
///   PlatformException(INFERENCE_ERROR, Invalid input name: attention_mask)
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

/// CLIP vision tower expects:
///   - pixel_values: float32[1, 3, 224, 224]
List<_SyntheticInput> _syntheticClipVisionInputs() {
  final width = 224, height = 224, channels = 3;
  final total = channels * height * width;
  final rng = math.Random(42);
  // CLIP expects normalized [-1, 1] range
  final pixelValues =
      List<double>.generate(total, (_) => rng.nextDouble() * 2 - 1);

  return [
    _SyntheticInput(
        name: 'pixel_values', shape: [1, channels, height, width], data: pixelValues),
  ];
}

/// PP-OCRv5 detection model input:
///   - x: float32[1, 3, 960, 960]  (variable size, 960 is common)
List<_SyntheticInput> _syntheticPpocrDetInputs() {
  final size = 960;
  final total = 3 * size * size;
  final rng = math.Random(42);
  // PP-OCR detection normalizes to [0, 1] typically, but uses mean=[0.485, 0.456, 0.406]
  // std=[0.229, 0.224, 0.225]. We use raw [0, 1] for validation.
  final pixelValues =
      List<double>.generate(total, (_) => rng.nextDouble());

  return [
    _SyntheticInput(name: 'x', shape: [1, 3, size, size], data: pixelValues),
  ];
}

/// PP-OCRv5 recognition model input:
///   - x: float32[1, 3, 48, 320]
List<_SyntheticInput> _syntheticPpocrRecInputs() {
  final h = 48, w = 320;
  final total = 3 * h * w;
  final rng = math.Random(42);
  final pixelValues =
      List<double>.generate(total, (_) => rng.nextDouble());

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

// ─── Shape helpers ──────────────────────────────────────────────────────────

bool _shapesMatch(List<int> actual, List<int> expected) {
  if (actual.length != expected.length) return false;
  for (var i = 0; i < actual.length; i++) {
    if (actual[i] != expected[i]) return false;
  }
  return true;
}
