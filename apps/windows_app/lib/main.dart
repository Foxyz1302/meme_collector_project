import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:forui/forui.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:signals_flutter/signals_flutter.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

import 'inference_factory.dart';
import 'inference_isolate.dart';
import 'ppocr_engine.dart';
import 'state/signals.dart';
import 'widgets/home_scaffold.dart';

/// Global flag for whether the Coordinator has finished initializing.
final isInitialized = signal<bool>(false);
final initError = signal<String?>(null);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Show the window immediately with a loading screen.
  // The Coordinator initializes in the background.
  runApp(const ReactionRouletteApp());

  // Initialize the Coordinator after the window is visible.
  // Use a microtask to ensure the window renders first.
  await Future.microtask(() {});
  await _initCoordinator();
}

Future<void> _initCoordinator() async {
  try {
    // ─── Determine storage path ────────────────────────────────────────────
    String storagePath;
    if (kReleaseMode) {
      final appSupport = await getApplicationSupportDirectory();
      storagePath = p.join(appSupport.path, 'reaction_roulette');
    } else {
      storagePath = p.join(Directory.current.path, '..', '..', 'storage');
    }
    await Directory(storagePath).create(recursive: true);

    // ─── Determine models directory ────────────────────────────────────────
    final modelsDir = p.normalize(p.join(Directory.current.path, '..', '..', 'assets', 'models'));

    // ─── Initialize inference isolate (runs ONNX off the main thread) ──────
    final isolateManager = InferenceIsolateManager();
    await isolateManager.spawn(
      InferenceIsolateConfig(
        clipTextModelPath: p.join(modelsDir, 'clip_text_fp16.onnx'),
        clipTokenizerPath: p.join(modelsDir, 'clip_tokenizer.json'),
        clipVisionModelPath: p.join(modelsDir, 'clip_vision_fp16.onnx'),
      ),
    );

    // ─── Initialize Coordinator ────────────────────────────────────────────
    coordinator = Coordinator(config: CoordinatorConfig(
      storagePath: storagePath,
      animatedPreviewsEnabled: true,
      ocrEnabled: false, // Set to true once OCR recognition is fixed
      ocrIsolateCallback: (imagePath) async {
        // Run OCR in a background isolate — creates its own ONNX sessions
        return await Isolate.run(() async {
          final engine = PpocrEngine(
            detModelPath: p.join(modelsDir, 'ppocr_det.onnx'),
            recModelPath: p.join(modelsDir, 'ppocr_rec.onnx'),
            dictPath: p.join(modelsDir, 'ppocr_dict.txt'),
          );
          await engine.init();
          final result = await engine.ocrFile(imagePath);
          await engine.dispose();
          return result;
        });
      },
    ));

    await coordinator!.init(AppInferenceFactory(modelsDir: modelsDir, isolateManager: isolateManager));

    // Wire coordinator streams to signals (for UI rebuilds)
    wireCoordinatorToSignals();

    // Initial refresh of reactions + hotlist
    refreshReactions();
    await refreshHotlist();

    // Scan for incomplete embeddings (after init, once metadata is loaded)
    scanIncomplete();

    // Mark as initialized — triggers UI rebuild from loading screen to main UI
    isInitialized.value = true;
  } catch (e, st) {
    initError.value = e.toString();
    debugPrint('Coordinator init failed: $e\n$st');
  }
}

class ReactionRouletteApp extends StatelessWidget {
  const ReactionRouletteApp({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = <TargetPlatform>{TargetPlatform.android, TargetPlatform.iOS, TargetPlatform.fuchsia}.contains(defaultTargetPlatform)
        ? FThemes.neutral.dark.touch
        : FThemes.neutral.dark.desktop;

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Reaction Roulette',
      supportedLocales: FLocalizations.supportedLocales,
      localizationsDelegates: const [...FLocalizations.localizationsDelegates],
      theme: theme.toApproximateMaterialTheme(),
      builder: (_, child) => FTheme(
        data: theme,
        child: FToaster(child: FTooltipGroup(child: child!)),
      ),
      home: SignalBuilder(
        builder: (context) {
          if (initError.value != null) {
            return _LoadingScreen(error: initError.value);
          }
          if (!isInitialized.value) {
            return const _LoadingScreen();
          }
          return const HomeScaffold();
        },
      ),
    );
  }
}

class _LoadingScreen extends StatelessWidget {
  final String? error;
  const _LoadingScreen({this.error});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (error != null) ...[
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text('Initialization failed', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(error!, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
              ),
            ] else ...[
              const SizedBox(width: 48, height: 48, child: CircularProgressIndicator()),
              const SizedBox(height: 24),
              Text('Loading Reaction Roulette...', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text('Initializing search engine + loading AI models', style: Theme.of(context).textTheme.bodySmall),
            ],
          ],
        ),
      ),
    );
  }
}
