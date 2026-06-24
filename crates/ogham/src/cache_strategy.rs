use ogham_core::{Message, meta_keys};

/// Provider cache behavior the host is targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Anthropic explicit breakpoints (max 4 per request).
    Anthropic,
    /// OpenAI: automatic prefix caching — alignment only, no annotations.
    OpenAi,
    /// Unknown provider: alignment only.
    Generic,
}

/// Replace provider cache breakpoints. Returns how many breakpoints were set.
///
/// Anthropic: set `metadata[CACHE_CONTROL] = "ephemeral"` on (in priority order,
/// skipping duplicates, max 4):
///   1. the LAST message with role == "system"
///   2. the message immediately BEFORE the last `stable_suffix` messages
///      (i.e. index len - stable_suffix - 1), if that index is > breakpoint 1's index
///
/// Existing CACHE_CONTROL keys are removed before the requested strategy is
/// applied. Non-Anthropic strategies remove all keys and return 0.
///
/// `stable_suffix` is the number of trailing messages expected to change every
/// turn (typically preserve_recent from ConversationConfig). Saturate at 0.
pub fn apply_cache_strategy(
    messages: &mut [Message],
    strategy: CacheStrategy,
    stable_suffix: usize,
) -> usize {
    match strategy {
        CacheStrategy::Anthropic => {
            for msg in messages.iter_mut() {
                msg.metadata.remove(meta_keys::CACHE_CONTROL);
            }

            let mut indices = Vec::new();

            // 1. the LAST message with role == "system"
            if let Some(idx) = messages.iter().rposition(|m| m.role == "system") {
                indices.push(idx);
            }

            // 2. the message immediately BEFORE the last stable_suffix messages.
            //    Only meaningful when a stable prefix exists at all
            //    (messages.len() > stable_suffix), and never duplicates or
            //    precedes the system breakpoint.
            if stable_suffix > 0 && messages.len() > stable_suffix {
                let idx2 = messages.len() - stable_suffix - 1;
                let after_first = indices.first().is_none_or(|&first| idx2 > first);
                if after_first && !indices.contains(&idx2) {
                    indices.push(idx2);
                }
            }

            let count = indices.len().min(4);
            for &idx in &indices[..count] {
                messages[idx].metadata.insert(
                    meta_keys::CACHE_CONTROL.to_string(),
                    "ephemeral".to_string(),
                );
            }
            count
        }
        CacheStrategy::OpenAi | CacheStrategy::Generic => {
            for msg in messages.iter_mut() {
                msg.metadata.remove(meta_keys::CACHE_CONTROL);
            }
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogham_core::Message;

    #[test]
    fn anthropic_sets_system_breakpoint() {
        let mut msgs = vec![
            Message::new("system", "You are helpful."),
            Message::new("user", "Hi"),
            Message::new("assistant", "Hello!"),
        ];
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 0);
        assert_eq!(n, 1);
        assert_eq!(
            msgs[0].metadata.get(meta_keys::CACHE_CONTROL),
            Some(&"ephemeral".to_string())
        );
    }

    #[test]
    fn anthropic_max_two_distinct() {
        let mut msgs: Vec<Message> = (0..10)
            .map(|i| {
                if i == 0 {
                    Message::new("system", "sys")
                } else if i % 2 == 1 {
                    Message::new("user", "u")
                } else {
                    Message::new("assistant", "a")
                }
            })
            .collect();
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 4);
        assert!(n <= 2);
        let marked: Vec<usize> = msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| m.metadata.contains_key(meta_keys::CACHE_CONTROL))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(marked.len(), n);
        assert!(marked.len() <= 4);
        // All marked indices must be distinct.
        let mut uniq = marked.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), marked.len());
    }

    #[test]
    fn anthropic_prefix_breakpoint_without_system() {
        // No system message: the stable-prefix breakpoint must still be set.
        let mut msgs: Vec<Message> = (0..6)
            .map(|i| Message::new(if i % 2 == 0 { "user" } else { "assistant" }, "m"))
            .collect();
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 2);
        assert_eq!(n, 1);
        assert!(msgs[3].metadata.contains_key(meta_keys::CACHE_CONTROL));
    }

    #[test]
    fn anthropic_all_volatile_sets_nothing() {
        // stable_suffix >= len: no stable prefix, nothing to annotate.
        let mut msgs: Vec<Message> = (0..3).map(|_| Message::new("user", "m")).collect();
        msgs[0].metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 3);
        assert_eq!(n, 0);
        assert!(
            msgs.iter()
                .all(|m| !m.metadata.contains_key(meta_keys::CACHE_CONTROL))
        );
    }

    #[test]
    fn anthropic_replaces_stale_annotations() {
        let mut msgs = vec![
            Message::new("system", "sys"),
            Message::new("user", "stable"),
            Message::new("assistant", "volatile"),
        ];
        msgs[2].metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );

        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 1);

        assert_eq!(n, 2);
        assert!(msgs[0].metadata.contains_key(meta_keys::CACHE_CONTROL));
        assert!(msgs[1].metadata.contains_key(meta_keys::CACHE_CONTROL));
        assert!(!msgs[2].metadata.contains_key(meta_keys::CACHE_CONTROL));
    }

    #[test]
    fn openai_strips_annotations() {
        let mut msgs = vec![Message::new("system", "sys"), Message::new("user", "hi")];
        msgs[0].metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::OpenAi, 0);
        assert_eq!(n, 0);
        assert!(!msgs[0].metadata.contains_key(meta_keys::CACHE_CONTROL));
        assert!(!msgs[1].metadata.contains_key(meta_keys::CACHE_CONTROL));
    }

    #[test]
    fn empty_messages_ok() {
        let mut msgs: Vec<Message> = vec![];
        let n = apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 0);
        assert_eq!(n, 0);
    }
}
