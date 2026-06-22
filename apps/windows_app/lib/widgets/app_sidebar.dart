import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:meme_collector_core/worker.dart';
import '../state/signals.dart';

class AppSidebar extends StatelessWidget {
  final String selected;
  const AppSidebar({super.key, required this.selected});

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    final typography = context.theme.typography;

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
                FAvatar.raw(child: Icon(FLucideIcons.userRound, size: 18, color: colors.mutedForeground)),
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
                        '0 items',
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
        FSidebarGroup(
          label: const Text('Library'),
          children: [
            FSidebarItem(
              icon: const Icon(FLucideIcons.layoutGrid),
              label: const Text('All'),
              selected: selected == 'all',
              onPress: () => selectedNav.value = 'all',
            ),
            FSidebarItem(
              icon: const Icon(FLucideIcons.clock),
              label: const Text('Recently Used'),
              selected: selected == 'recent',
              onPress: () => selectedNav.value = 'recent',
            ),
            FSidebarItem(
              icon: const Icon(FLucideIcons.tags),
              label: const Text('Tags'),
              selected: selected == 'tags',
              onPress: () => selectedNav.value = 'tags',
            ),
            FSidebarItem(
              icon: const Icon(FLucideIcons.link),
              label: const Text('Sources'),
              selected: selected == 'sources',
              onPress: () => selectedNav.value = 'sources',
            ),
          ],
        ),
        FSidebarGroup(
          label: const Text('Add'),
          action: const Icon(FLucideIcons.plus),
          onActionPress: () {
            // TODO: open import dialog (paste URL / pick file)
          },
          children: [
            FSidebarItem(
              icon: const Icon(FLucideIcons.clipboardPaste),
              label: const Text('Paste link'),
              onPress: () async {
                final result = (await Clipboard.getData('text/plain'))?.text;
                if (result == null) {
                  return;
                }
                coordinator?.addReaction(url: (result).trim());
                // TODO: clipboard -> parse tenor/giphy -> download
              },
            ),
            FSidebarItem(
              icon: const Icon(FLucideIcons.fileUp),
              label: const Text('Import file'),
              onPress: () {
                // TODO: file_picker -> copy into library dir
              },
            ),
          ],
        ),
      ],
    );
  }
}
