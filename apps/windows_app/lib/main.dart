import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:forui/forui.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

import 'inference_factory.dart';
import 'state/signals.dart';
import 'widgets/home_scaffold.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // ─── Determine storage path ────────────────────────────────────────────
  // For dev mode: use ./storage/ next to the app exe.
  // TODO: replace with first-launch storage wizard (config.json + picker)
  String storagePath;
  if (kReleaseMode) {
    // In release: use App Support directory
    final appSupport = await getApplicationSupportDirectory();
    storagePath = p.join(appSupport.path, 'reaction_roulette');
  } else {
    // In debug: use ./storage/ next to the project (easy to inspect)
    storagePath = p.join(Directory.current.path, '..', '..', 'storage');
  }
  await Directory(storagePath).create(recursive: true);

  // ─── Determine models directory ────────────────────────────────────────
  // Models live at <workspace_root>/assets/models/
  final modelsDir = p.normalize(
      p.join(Directory.current.path, '..', '..', 'assets', 'models'));

  // ─── Initialize Coordinator ────────────────────────────────────────────
  coordinator = Coordinator(
    config: CoordinatorConfig(
      storagePath: storagePath,
      animatedPreviewsEnabled: false, // TODO: settings toggle
      ocrEnabled: false, // TODO: implement PpocrEngine first
    ),
  );

  await coordinator!.init(AppInferenceFactory(modelsDir: modelsDir));

  // Wire coordinator streams to signals (for UI rebuilds)
  wireCoordinatorToSignals();

  // Initial refresh of reactions + hotlist
  refreshReactions();
  await refreshHotlist();

  runApp(const ReactionRouletteApp());
}

class ReactionRouletteApp extends StatelessWidget {
  const ReactionRouletteApp({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = <TargetPlatform>{
      TargetPlatform.android,
      TargetPlatform.iOS,
      TargetPlatform.fuchsia,
    }.contains(defaultTargetPlatform)
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
      home: const HomeScaffold(),
    );
  }
}
