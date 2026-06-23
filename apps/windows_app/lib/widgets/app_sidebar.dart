import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../state/signals.dart';

class AppSidebar extends StatelessWidget {
  final String selected;
  const AppSidebar({super.key, required this.selected});

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    final typography = context.theme.typography;

    // Count of currently ingesting reactions (for showing progress)
    final ingestingCount = ingestStatus.value.values.where((s) => s.isNotEmpty && !s.startsWith('Failed')).length;

    return FSidebar(
      header: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
        child: Text('Reaction Roulette', style: typography.body.sm.copyWith(fontWeight: FontWeight.bold)),
      ),
      footer: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: FCard.raw(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              spacing: 10,
              children: [
                // FAvatar.raw(child: Icon(FLucideIcons.userRound, size: 18, color: colors.mutedForeground)),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    spacing: 2,
                    children: [
                      Text(
                        'Local library',
                        overflow: TextOverflow.ellipsis,
                        style: typography.body.sm.copyWith(color: colors.foreground),
                      ),
                      Text(
                        '${allReactions.value.length} items',
                        overflow: TextOverflow.ellipsis,
                        style: typography.body.xs.copyWith(color: colors.mutedForeground),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      children: [
        // ─── Incomplete embeddings notice ───────────────────────────────
        // Shows when reactions are missing embeddings. Stays visible until
        // all reactions are fully embedded. Has a "Process" button.
        SignalBuilder(
          builder: (context) {
            final incomplete = incompleteCount.value;
            final ingesting = ingestingCount;
            if (incomplete == 0 && ingesting == 0) return const SizedBox.shrink();

            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: FCard.raw(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    spacing: 8,
                    children: [
                      Row(
                        spacing: 8,
                        children: [
                          Icon(ingesting > 0 ? FLucideIcons.loader : FLucideIcons.alertTriangle, size: 16, color: colors.primary),
                          Expanded(
                            child: Text(
                              ingesting > 0 ? 'Embedding $ingesting reactions…' : '$incomplete reactions need embedding',
                              style: typography.body.xs.copyWith(color: colors.foreground, fontWeight: FontWeight.w600),
                            ),
                          ),
                        ],
                      ),
                      if (incomplete > 0 && ingesting == 0)
                        FButton(
                          size: .xs,
                          mainAxisSize: MainAxisSize.max,
                          onPress: () {
                            final count = processIncomplete();
                            if (count > 0 && context.mounted) {
                              showFToast(context: context, title: Text('Processing $count reactions…'));
                            }
                          },
                          child: const Text('Process now'),
                        ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),

        // ─── Library ────────────────────────────────────────────────────
        // FSidebarGroup(
        //   label: const Text('Library'),
        //   children: [
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.layoutGrid),
        //       label: const Text('All'),
        //       selected: selected == 'all',
        //       onPress: () => selectedNav.value = 'all',
        //     ),
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.clock),
        //       label: const Text('Recently Used'),
        //       selected: selected == 'recent',
        //       onPress: () => selectedNav.value = 'recent',
        //     ),
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.tags),
        //       label: const Text('Tags'),
        //       selected: selected == 'tags',
        //       onPress: () => selectedNav.value = 'tags',
        //     ),
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.link),
        //       label: const Text('Sources'),
        //       selected: selected == 'sources',
        //       onPress: () => selectedNav.value = 'sources',
        //     ),
        //   ],
        // ),

        // ─── Add ────────────────────────────────────────────────────────
        // FSidebarGroup(
        //   label: const Text('Add'),
        //   action: const Icon(FLucideIcons.plus),
        //   onActionPress: () {
        //     // TODO: open import dialog (paste URL / pick file)
        //   },
        //   children: [
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.clipboardPaste),
        //       label: const Text('Paste link'),
        //       onPress: () async {
        //         final result = (await Clipboard.getData('text/plain'))?.text;
        //         if (result == null) {
        //           return;
        //         }
        //         await addReaction(result.trim());
        //       },
        //     ),
        //     FSidebarItem(
        //       icon: const Icon(FLucideIcons.fileUp),
        //       label: const Text('Import file'),

        //       onPress: () {
        //         // TODO: file_picker -> copy into library dir
        //       },
        //     ),
        //   ],
        // ),

        // ─── Maintenance ────────────────────────────────────────────────
        FSidebarGroup(
          label: const Text('Maintenance'),
          children: [
            FSidebarItem(
              icon: const Icon(FLucideIcons.refreshCw),
              label: const Text('Re-embed all'),
              onPress: () {
                final count = resetAll();
                if (count > 0 && context.mounted) {
                  showFToast(context: context, title: Text('Re-embedding $count reactions…'));
                }
              },
            ),
            FSidebarItem(
              icon: const Icon(FLucideIcons.scanLine),
              label: const Text('Scan for incomplete'),
              onPress: () {
                final count = scanIncomplete();
                if (context.mounted) {
                  showFToast(context: context, title: Text(count > 0 ? '$count reactions need embedding' : 'All reactions are complete'));
                }
              },
            ),
          ],
        ),
      ],
    );
  }
}
