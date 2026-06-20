import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:forui/forui.dart';

import 'widgets/home_scaffold.dart';

void main() => runApp(const ReactionRouletteApp());

class ReactionRouletteApp extends StatelessWidget {
  const ReactionRouletteApp({super.key});

  @override
  Widget build(BuildContext context) {
    // defaultTargetPlatform is not a compile-time constant, so the set
    // cannot be `const`. android/iOS/fuchsia are enum values on TargetPlatform.
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
