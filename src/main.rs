use clap::{Parser, ValueEnum};
use ordered_float::OrderedFloat;
use rayon::slice::ParallelSliceMut;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{self, BufRead};

type Token = u32; // u16;
type Index = u32;

const DEFAULT_MIN_NGRAM_SIZE: usize = 2;
const DEFAULT_MAX_NGRAM_SIZE: usize = 15;
const DEFAULT_MAX_HEAP_SIZE: usize = 2000;
const DEFAULT_MIN_COUNT: usize = 10;
const DEFAULT_MAX_SOURCE_TAGS: usize = 0;
const MIN_POSITIVE_PMI: f64 = 0.1;

/// A consecutive block of TSV rows from the same source tag.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceSpan {
    source_id: usize,
    tag: String,
    token_start: usize,
    token_end: usize,
}

/// Tokenized corpus data plus compact source provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Corpus {
    tokens: Vec<Token>,
    token_decoder: Vec<String>,
    token_occurrences: Vec<usize>,
    sources: Vec<SourceSpan>,
}

/// Convert a normalized word to a token, creating a new token when needed.
fn token_for_word(
    token_map: &mut HashMap<String, Token>,
    token_decoder: &mut Vec<String>,
    word_with_punct: &str,
) -> Option<Token> {
    // Check the raw token first so already-normalized entries avoid the
    // trimming and lowercasing work entirely.
    if let Some(&token) = token_map.get(word_with_punct) {
        return Some(token);
    }

    let trimmed_word =
        word_with_punct.trim_matches(|c: char| c.is_ascii_punctuation() || c.is_ascii_control());

    if trimmed_word.is_empty() {
        return None;
    }

    let normalized_word = trimmed_word.to_lowercase();
    if let Some(&token) = token_map.get(normalized_word.as_str()) {
        Some(token)
    } else {
        if token_decoder.len() == Token::MAX as usize {
            panic!("Too many unique tokens (max: {})", Token::MAX);
        }
        let new_token = token_decoder.len() as Token;
        token_map.insert(normalized_word.clone(), new_token);
        token_decoder.push(normalized_word);
        Some(new_token)
    }
}

/// Convert two-column TSV sentence rows to token IDs and source spans.
///
/// The first TSV column is a source tag, and the second column is the sentence.
/// End-of-line (`<EOL>`) is assigned token value 0 after every sentence row.
fn tokenize_tsv_reader<R: BufRead>(mut reader: R) -> Result<Corpus, String> {
    eprintln!("Tokenizing file");

    let mut token_map = HashMap::new();
    let mut token_decoder = vec!["<EOL>".to_string()];
    let mut token_occurrences = vec![0usize];
    let mut token_vec = Vec::new();
    let mut sources: Vec<SourceSpan> = Vec::new();
    let mut line = String::new();
    let mut line_number = 1usize;
    let mut current_tag: Option<String> = None;
    let mut current_source_index: Option<usize> = None;

    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|error| format!("Failed to read line {line_number}: {error}"))?;
        if bytes_read == 0 {
            break;
        }

        let Some((source_tag, sentence)) = line.split_once('\t') else {
            return Err(format!(
                "Malformed TSV row {line_number}: missing tab separator"
            ));
        };
        if source_tag.is_empty() {
            return Err(format!("Malformed TSV row {line_number}: empty source tag"));
        }

        if current_tag.as_deref() != Some(source_tag) {
            if let Some(source_index) = current_source_index {
                sources[source_index].token_end = token_vec.len();
            }

            current_tag = Some(source_tag.to_string());
            current_source_index = Some(sources.len());
            sources.push(SourceSpan {
                source_id: sources.len(),
                tag: source_tag.to_string(),
                token_start: token_vec.len(),
                token_end: token_vec.len(),
            });
        }

        for word_with_punct in sentence.split_whitespace() {
            let Some(token) = token_for_word(&mut token_map, &mut token_decoder, word_with_punct)
            else {
                continue;
            };

            if token as usize == token_occurrences.len() {
                token_occurrences.push(0);
            }

            token_vec.push(token);
            token_occurrences[token as usize] += 1;
        }

        token_vec.push(0);
        token_occurrences[0] += 1;

        if (line_number - 1).is_multiple_of(1000000) {
            eprintln!("Processed {} lines", line_number);
        }
        line_number += 1;
    }

    if let Some(source_index) = current_source_index {
        sources[source_index].token_end = token_vec.len();
    }

    eprintln!(
        "Processed {} tokens, {} distinct",
        token_vec.len(),
        token_decoder.len()
    );
    eprintln!("Counted {} tokens", token_occurrences.len());

    Ok(Corpus {
        tokens: token_vec,
        token_decoder,
        token_occurrences,
        sources,
    })
}

/// Convert each TSV sentence row in a file to token values and source spans.
fn tokenize_tsv_file(path: &str) -> Result<Corpus, String> {
    let file = File::open(path).map_err(|error| format!("Failed to open {path}: {error}"))?;
    let reader = io::BufReader::new(file);
    tokenize_tsv_reader(reader)
}

/// Ranking methods available for scored n-gram candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RankBy {
    /// Rank n-grams by the MI2 association score.
    Mi2,
    /// Rank n-grams by the Dice association score.
    Dice,
    /// Rank n-grams by their raw occurrence count.
    Frequency,
    /// Rank n-grams by their token length.
    #[value(alias = "longest")]
    Length,
}

/// Command line arguments for the n-gram ranking tool.
#[derive(Parser)]
#[command(about = "Text search tool using suffix arrays", version)]
struct Args {
    /// Two-column TSV file to search in
    #[arg(required = true)]
    file: String,

    /// Metric used to rank n-gram candidates
    #[arg(short = 'r', long, value_enum, default_value = "mi2")]
    rank_by: RankBy,

    /// Smallest n-gram size to score
    #[arg(short = 'n', long, default_value_t = DEFAULT_MIN_NGRAM_SIZE)]
    min_ngram_size: usize,

    /// Largest n-gram size to score
    #[arg(short = 'x', long, default_value_t = DEFAULT_MAX_NGRAM_SIZE)]
    max_ngram_size: usize,

    /// Maximum number of scored n-grams to retain; 0 keeps every scored n-gram
    #[arg(short = 'k', long, default_value_t = DEFAULT_MAX_HEAP_SIZE)]
    max_heap_size: usize,

    /// Minimum repeated n-gram occurrence count to score; must be at least 2
    #[arg(short = 'c', long, default_value_t = DEFAULT_MIN_COUNT)]
    min_count: usize,

    /// Maximum number of source tags to print for each n-gram; 0 disables source output
    #[arg(short = 's', long, default_value_t = DEFAULT_MAX_SOURCE_TAGS)]
    max_source_tags: usize,
}

impl Args {
    /// Validate cross-field CLI constraints that clap cannot express directly.
    fn validate(&self) -> Result<(), String> {
        if self.min_ngram_size == 0 {
            return Err("--min-ngram-size must be at least 1".to_string());
        }
        if self.max_ngram_size < self.min_ngram_size {
            return Err(format!(
                "--max-ngram-size ({}) must be greater than or equal to --min-ngram-size ({})",
                self.max_ngram_size, self.min_ngram_size
            ));
        }
        if self.min_count < 2 {
            return Err(
                "--min-count must be at least 2 because enumeration is repeated-only".to_string(),
            );
        }
        Ok(())
    }

    /// Parse arguments and exit with a clear error if the size bounds are invalid.
    fn parse_validated() -> Args {
        let args = Args::parse();
        if let Err(message) = args.validate() {
            eprintln!("error: {message}");
            std::process::exit(2);
        }
        args
    }
}

/// Populates a suffix array from a vector of tokens
///
/// # Arguments
/// * `tokens` - The vector of tokens to build the suffix array from
///
/// # Returns
/// A vector of indices into the tokens array, sorted by suffix order
fn build_suffix_array(tokens: &[Token]) -> Vec<Index> {
    eprintln!("Building suffix array");

    // Initialize with all suffix indices: 0, 1, 2, ..., tokens.len()-1
    let mut sarray: Vec<Index> = (0..tokens.len() as Index).collect();

    // Sort indices by lexicographical order of their corresponding suffixes
    // sarray.sort_by_key(|&u| &tokens[u as usize..]);
    sarray.par_sort_unstable_by_key(|&u| &tokens[u as usize..]);

    eprintln!("Built suffix array");
    sarray
}

/// Build an LCP (Longest Common Prefix) array for the suffix array using Kasai's algorithm
///
/// The LCP array stores the length of the longest common prefix between consecutive suffixes
/// in the sorted suffix array. For example, if suffix_array[i] and suffix_array[i+1] share
/// the first 3 tokens, then lcp[i] = 3.
///
/// This implementation has a special rule: EOL tokens (token value 0) are treated as unequal,
/// meaning the LCP stops when it encounters an EOL token.
///
/// # Arguments
/// * `sarray` - The suffix array (sorted indices into the tokens array)
/// * `tokens` - The original token vector
///
/// # Returns
/// The LCP array where lcp[i] is the longest common prefix length between
/// the suffixes at sarray[i] and sarray[i+1].
fn build_lcp_array(sarray: &[Index], tokens: &[Token]) -> Vec<u8> {
    eprintln!("Building LCP array");

    // Store LCP values compactly; current corpora have maximum values under
    // u8::MAX, and the assignment below checks that assumption.
    let mut lcp: Vec<u8> = vec![0; sarray.len()];

    // Build the inverse suffix array: inv_sarray[i] tells us the position in sarray
    // where suffix i appears. This allows us to quickly find a suffix's position
    // in the sorted order.
    let mut inv_sarray: Vec<Index> = vec![0; sarray.len()];
    for (i, &s) in sarray.iter().enumerate() {
        inv_sarray[s as usize] = i as Index;
    }

    // h tracks the LCP length from the previous iteration
    // Kasai's algorithm exploits the fact that if suffix i has LCP h with its successor,
    // then suffix i+1 has LCP at least h-1 with its successor (this is the key insight
    // that makes the algorithm O(n) instead of O(n²))
    let mut h: usize = 0;

    // Iterate through suffixes in text order (not sorted order)
    for i in 0..sarray.len() {
        // If this suffix is the last one in sorted order, it has no successor to compare with
        if inv_sarray[i] as usize == sarray.len() - 1 {
            h = 0;
            continue;
        }

        // Find the next suffix in sorted order after suffix i
        let j = sarray[inv_sarray[i] as usize + 1] as usize;

        // Extend the common prefix as far as possible
        // Stop if we reach the end of either suffix, or if tokens differ,
        // or if we encounter an EOL token (0) in either suffix
        while i + h < sarray.len()
            && j + h < sarray.len()
            && tokens[i + h] == tokens[j + h]
            && tokens[i + h] != 0
            && tokens[j + h] != 0
        {
            h += 1;
        }

        // Store the LCP value at the position of suffix i in the sorted array
        lcp[inv_sarray[i] as usize] = u8::try_from(h).expect("LCP value exceeded u8::MAX");

        // Decrease h by 1 for the next iteration (Kasai's optimization)
        // saturating_sub ensures we don't go below 0
        h = h.saturating_sub(1);
    }

    eprintln!("Built LCP array");
    // eprintln!("LCP array: {:?}", lcp);
    lcp
}

/// Return the largest prefix length stored in an LCP array.
fn max_lcp_value(lcp: &[u8]) -> u8 {
    // Empty inputs occur only for degenerate callers, but returning zero keeps
    // the reporting path total and avoids special cases at the call site.
    lcp.iter().copied().max().unwrap_or(0)
}

/// Function signature shared by all n-gram scoring methods.
type ScoringFunction = fn(&[Token], usize, &[usize], usize) -> Option<f64>;

/// An n-gram candidate that has been scored for ranking and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScoredNgram {
    score: OrderedFloat<f64>,
    token_start: Index,
    suffix_index: Index,
    len: Index,
    left: Index,
    right: Index,
}

impl ScoredNgram {
    #[cfg(test)]
    fn new(score: f64, token_start: Index, suffix_index: Index, len: Index) -> ScoredNgram {
        ScoredNgram::with_interval(
            score,
            token_start,
            suffix_index,
            len,
            suffix_index,
            suffix_index,
        )
    }

    fn with_interval(
        score: f64,
        token_start: Index,
        suffix_index: Index,
        len: Index,
        left: Index,
        right: Index,
    ) -> ScoredNgram {
        ScoredNgram {
            score: OrderedFloat(score),
            token_start,
            suffix_index,
            len,
            left,
            right,
        }
    }
}

impl Ord for ScoredNgram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher scores rank first. Ties prefer longer n-grams, then earlier
        // source positions, then earlier suffix-array positions for stability.
        self.score
            .cmp(&other.score)
            .then_with(|| self.len.cmp(&other.len))
            .then_with(|| other.token_start.cmp(&self.token_start))
            .then_with(|| other.suffix_index.cmp(&self.suffix_index))
    }
}

impl PartialOrd for ScoredNgram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A heap that either keeps all scored n-grams or only the best scored n-grams.
struct ScoredNgramHeap {
    max_size: Option<usize>,
    heap: BinaryHeap<Reverse<ScoredNgram>>,
}

impl ScoredNgramHeap {
    fn new(max_size: usize) -> ScoredNgramHeap {
        ScoredNgramHeap {
            max_size: (max_size > 0).then_some(max_size),
            heap: BinaryHeap::new(),
        }
    }

    fn push(&mut self, scored_ngram: ScoredNgram) {
        // With no size limit, keep every scored n-gram for final sorting.
        let Some(max_size) = self.max_size else {
            self.heap.push(Reverse(scored_ngram));
            return;
        };

        // With a size limit, retain the highest-scoring candidates without
        // sorting the full stream.
        if self.heap.len() < max_size {
            self.heap.push(Reverse(scored_ngram));
        } else if scored_ngram > self.heap.peek().unwrap().0 {
            self.heap.pop();
            self.heap.push(Reverse(scored_ngram));
        }
    }

    fn into_vec(self) -> Vec<ScoredNgram> {
        self.heap.into_iter().map(|wrapped| wrapped.0).collect()
    }
}

/// Sort retained n-grams by score, then deterministic tie-breakers for output.
fn sort_scored_ngrams_for_output(scored_ngrams: &mut [ScoredNgram]) {
    scored_ngrams.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.len.cmp(&a.len))
            .then_with(|| a.token_start.cmp(&b.token_start))
            .then_with(|| a.suffix_index.cmp(&b.suffix_index))
    })
}

// Debugging output for a ScoredNgramHeap.
impl std::fmt::Debug for ScoredNgramHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScoredNgramHeap")
            .field("max_size", &self.max_size)
            .field("heap", &self.heap)
            .finish()
    }
}

/// Choose the requested scoring function while keeping metric implementations separate.
fn scoring_function_for(rank_by: RankBy) -> ScoringFunction {
    match rank_by {
        RankBy::Mi2 => score_mi2,
        RankBy::Dice => score_dice,
        RankBy::Frequency => score_frequency,
        RankBy::Length => score_length,
    }
}

/// Return whether an interval can emit only its longest owned n-gram.
fn rank_uses_longest_owned_length_only(rank_by: RankBy) -> bool {
    match rank_by {
        // Every owned length in a suffix-array interval has the same frequency,
        // and output tie-breaks prefer the longest n-gram.
        RankBy::Frequency | RankBy::Length => true,
        // For MI2, extending a same-count n-gram by token x changes PMI by
        // ln(corpus_token_count / token_occurrences[x]), which is nonnegative.
        // Since MI2 scores only positive-PMI candidates, the score cannot
        // decrease inside one same-count interval.
        RankBy::Mi2 => true,
        RankBy::Dice => false,
    }
}

/// Calculate PMI and apply the shared positive-PMI threshold used by affinity scores.
fn positive_pmi(
    ngram_tokens: &[Token],
    ngram_occurrences: usize,
    token_occurrences: &[usize],
    corpus_token_count: usize,
) -> Option<f64> {
    // The numerator is the n-gram occurrence count times the corpus token count
    // raised to the number of joins between words in the n-gram.
    let numerator = (ngram_occurrences as f64)
        * (corpus_token_count as f64).powf((ngram_tokens.len() - 1) as f64);

    // The denominator is the product of each token's corpus occurrence count.
    let denominator = ngram_tokens
        .iter()
        .map(|&t| token_occurrences[t as usize] as f64)
        .product::<f64>();

    // PMI is the natural log of the numerator/denominator ratio.
    let pmi = numerator.ln() - denominator.ln();
    if pmi > MIN_POSITIVE_PMI {
        Some(pmi)
    } else {
        None
    }
}

/// Score an n-gram with the MI2 association score.
fn score_mi2(
    ngram_tokens: &[Token],
    ngram_occurrences: usize,
    token_occurrences: &[usize],
    corpus_token_count: usize,
) -> Option<f64> {
    let pmi = positive_pmi(
        ngram_tokens,
        ngram_occurrences,
        token_occurrences,
        corpus_token_count,
    )?;
    Some(ngram_occurrences as f64 * pmi * pmi)
}

/// Score an n-gram with the Dice association score.
fn score_dice(
    ngram_tokens: &[Token],
    ngram_occurrences: usize,
    token_occurrences: &[usize],
    corpus_token_count: usize,
) -> Option<f64> {
    positive_pmi(
        ngram_tokens,
        ngram_occurrences,
        token_occurrences,
        corpus_token_count,
    )?;

    // Dice compares the n-gram occurrence count with the sum of member-token counts.
    let member_token_occurrences = ngram_tokens
        .iter()
        .map(|&t| token_occurrences[t as usize] as f64)
        .sum::<f64>();
    Some((ngram_tokens.len() as f64) * (ngram_occurrences as f64) / member_token_occurrences)
}

/// Score an n-gram by raw frequency.
fn score_frequency(
    _ngram_tokens: &[Token],
    ngram_occurrences: usize,
    _token_occurrences: &[usize],
    _corpus_token_count: usize,
) -> Option<f64> {
    Some(ngram_occurrences as f64)
}

/// Score an n-gram by its token length.
fn score_length(
    ngram_tokens: &[Token],
    _ngram_occurrences: usize,
    _token_occurrences: &[usize],
    _corpus_token_count: usize,
) -> Option<f64> {
    Some(ngram_tokens.len() as f64)
}

/// Enumerate repeated n-gram candidates, score them, and retain the best matches.
///
/// This function uses a monotone stack over the LCP array to emit repeated suffix-array
/// interval classes exactly once. It intentionally does not enumerate singleton n-grams.
fn collect_top_scored_ngrams(
    sarray: &[Index],
    lcp: &[u8],
    tokens: &[Token],
    token_occurrences: &[usize],
    min_ngram_size: usize,
    max_ngram_size: usize,
    max_heap_size: usize,
    min_count: usize,
    rank_by: RankBy,
) -> Vec<ScoredNgram> {
    eprintln!("Scoring ngrams");

    debug_assert!(
        min_count >= 2,
        "stack interval enumeration is repeated-only"
    );

    // corpus_token_count is the denominator scale used by PMI-based scores.
    let corpus_token_count = token_occurrences.iter().sum::<usize>();

    // best_ngrams owns the ranking policy: either retain every scored candidate
    // or retain only the highest-scoring fixed-size heap.
    let mut best_ngrams = ScoredNgramHeap::new(max_heap_size);
    let scoring_function = scoring_function_for(rank_by);
    let longest_owned_length_only = rank_uses_longest_owned_length_only(rank_by);

    // stack stores active LCP intervals as (left, prefix_len).
    //
    // Stack invariant at the start of each boundary iteration:
    // - Prefix lengths are strictly increasing from bottom to top.
    // - Each entry (left, prefix_len) represents suffix-array positions
    //   left..=boundary sharing the same prefix_len-token prefix.
    // - left is the earliest suffix-array position from which that prefix is
    //   shared by every suffix through boundary.
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for boundary in 0..sarray.len() {
        // boundary is the right edge currently being processed. LCP[boundary]
        // compares suffix-array positions boundary and boundary + 1. The final
        // LCP value is the zero sentinel that flushes all active intervals.
        let boundary_lcp = lcp[boundary] as usize;

        // interval_start is the left edge to use if boundary_lcp starts a new
        // active interval. It moves left when taller intervals are popped, so
        // shorter prefixes inherit the full interval they can still cover.
        let mut interval_start = boundary;

        // Pop every active prefix that cannot cross the current boundary.
        //
        // Loop invariant:
        // - Before each pop, all stack entries below the top still represent
        //   valid active intervals ending at this boundary.
        // - interval_start is the leftmost position inherited from intervals
        //   popped so far at this boundary.
        while stack
            .last()
            .is_some_and(|&(_, prefix_len)| prefix_len > boundary_lcp)
        {
            // left and prefix_len describe the interval that just ended:
            // suffixes left..=right share prefix_len tokens, and no suffix
            // just outside the interval can extend that same prefix.
            let (left, prefix_len) = stack.pop().unwrap();

            // right is the inclusive right endpoint of the popped suffix-array
            // interval. Its suffix is used as the representative n-gram source.
            let right = boundary;

            // ngram_occurrences is the number of suffixes in the interval, so
            // it is the occurrence count for every owned length in this class.
            let ngram_occurrences = right - left + 1;

            // parent_len is the longest prefix length owned by an enclosing
            // interval. The popped interval owns only lengths greater than this.
            let mut parent_len = boundary_lcp;
            if let Some(&(_, enclosing_prefix_len)) = stack.last() {
                parent_len = parent_len.max(enclosing_prefix_len);
            }

            // Low-frequency interval classes cannot contribute scored n-grams.
            if ngram_occurrences >= min_count {
                // first_len and last_len bound the concrete n-gram lengths
                // owned by this interval after applying caller limits.
                let first_len = (parent_len + 1).max(min_ngram_size);
                let last_len = prefix_len.min(max_ngram_size);

                if first_len <= last_len {
                    // suffix_start is the corpus token position for the
                    // representative suffix at the interval's right endpoint.
                    let suffix_start = sarray[right] as usize;

                    let first_emitted_len = if longest_owned_length_only {
                        last_len
                    } else {
                        first_len
                    };

                    for len in first_emitted_len..=last_len {
                        // ngram_tokens is valid without an EOL check because
                        // LCP construction stops before crossing EOL.
                        let ngram_tokens = &tokens[suffix_start..suffix_start + len];

                        if let Some(score) = scoring_function(
                            ngram_tokens,
                            ngram_occurrences,
                            token_occurrences,
                            corpus_token_count,
                        ) {
                            best_ngrams.push(ScoredNgram::with_interval(
                                score,
                                suffix_start as Index,
                                right as Index,
                                len as Index,
                                left as Index,
                                right as Index,
                            ));
                        }
                    }
                }
            }

            interval_start = left;
        }

        // Push boundary_lcp if it starts a new positive active interval.
        //
        // Push invariant:
        // - boundary_lcp == 0 starts no repeated prefix.
        // - If boundary_lcp is already represented by the current stack top,
        //   the existing interval simply continues.
        // - If it is larger than the current stack top, interval_start is the
        //   earliest left endpoint for that new shared prefix.
        if boundary_lcp > 0
            && stack
                .last()
                .is_none_or(|&(_, prefix_len)| prefix_len < boundary_lcp)
        {
            stack.push((interval_start, boundary_lcp));
        }
    }

    let mut scored_ngrams = best_ngrams.into_vec();
    sort_scored_ngrams_for_output(&mut scored_ngrams);
    scored_ngrams
}

/// Find the source span containing a token position.
fn source_index_for_token(sources: &[SourceSpan], token_start: usize) -> Option<usize> {
    let source_index = sources.partition_point(|source| source.token_start <= token_start);
    let source_index = source_index.checked_sub(1)?;
    let source = &sources[source_index];

    (token_start < source.token_end).then_some(source_index)
}

/// Return up to `max_source_tags` source tags for an n-gram occurrence interval.
fn source_tags_for_ngram<'a>(
    scored_ngram: &ScoredNgram,
    sarray: &[Index],
    sources: &'a [SourceSpan],
    max_source_tags: usize,
) -> Vec<&'a str> {
    if max_source_tags == 0 {
        return Vec::new();
    }

    let mut source_tags = Vec::new();
    let mut seen_source_ids = Vec::new();
    let left = scored_ngram.left as usize;
    let right = scored_ngram.right as usize;

    for &token_start in &sarray[left..=right] {
        let Some(source_index) = source_index_for_token(sources, token_start as usize) else {
            continue;
        };
        let source = &sources[source_index];

        if seen_source_ids.contains(&source.source_id) {
            continue;
        }

        seen_source_ids.push(source.source_id);
        source_tags.push(source.tag.as_str());
        if source_tags.len() == max_source_tags {
            break;
        }
    }

    source_tags
}

/// Render one scored n-gram line.
fn format_scored_ngram(
    scored_ngram: &ScoredNgram,
    tokens: &[Token],
    token_decoder: &[String],
    sarray: &[Index],
    sources: &[SourceSpan],
    max_source_tags: usize,
) -> String {
    let suffix_start = scored_ngram.token_start as usize;
    let suffix_len = scored_ngram.len as usize;
    let ngram_tokens = &tokens[suffix_start..suffix_start + suffix_len];
    let words: Vec<&str> = ngram_tokens
        .iter()
        .map(|&t| token_decoder[t as usize].as_str())
        .collect();
    let mut line = format!("{} ({})", words.join(" "), scored_ngram.score.0);

    if max_source_tags > 0 {
        let source_tags =
            source_tags_for_ngram(scored_ngram, sarray, sources, max_source_tags).join(", ");
        line.push_str(format!("\t[{source_tags}]").as_str());
    }

    line
}

/// Print scored n-grams in their already-ranked order.
fn print_scored_ngrams(
    scored_ngrams: &[ScoredNgram],
    tokens: &[Token],
    token_decoder: &[String],
    sarray: &[Index],
    sources: &[SourceSpan],
    max_source_tags: usize,
) {
    for scored_ngram in scored_ngrams {
        println!(
            "{}",
            format_scored_ngram(
                scored_ngram,
                tokens,
                token_decoder,
                sarray,
                sources,
                max_source_tags
            )
        );
    }
}

fn main() {
    let args = Args::parse_validated();
    let path = &args.file;

    let corpus = match tokenize_tsv_file(path) {
        Ok(corpus) => corpus,
        Err(message) => {
            eprintln!("error: {message}");
            std::process::exit(1);
        }
    };
    let tokens = &corpus.tokens;
    let token_decoder = &corpus.token_decoder;
    let token_occurrences = &corpus.token_occurrences;
    // eprintln!("{:?} tokens", tokens);
    // eprintln!("{:?} token_decoder", token_decoder);
    // eprintln!("Token occurrences: {:?}", token_occurrences);

    let sarray = build_suffix_array(tokens);

    // eprintln!("Suffix array: {:?}", sarray);
    // for i in 0..sarray.len() {
    //     let suffix_start = sarray[i] as usize;
    //     let suffix = &tokens[suffix_start..];
    //     eprint!("{}: ", i);
    //     for token in suffix {
    //         eprint!("{} ", token_decoder[*token as usize]);
    //     }
    //     eprintln!("");
    // }

    let lcp = build_lcp_array(&sarray, tokens);
    eprintln!("Maximum LCP value: {}", max_lcp_value(&lcp));

    // for i in 0..sarray.len() {
    //     println!("{} {} {} {}", sarray[i], tokens[sarray[i] as usize],token_decoder[tokens[sarray[i] as usize] as usize], lcp[i]);
    // }
    let scored_ngrams = collect_top_scored_ngrams(
        &sarray,
        &lcp,
        tokens,
        token_occurrences,
        args.min_ngram_size,
        args.max_ngram_size,
        args.max_heap_size,
        args.min_count,
        args.rank_by,
    );
    print_scored_ngrams(
        &scored_ngrams,
        tokens,
        token_decoder,
        &sarray,
        &corpus.sources,
        args.max_source_tags,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        Args, RankBy, ScoredNgram, ScoredNgramHeap, SourceSpan, build_lcp_array,
        build_suffix_array, collect_top_scored_ngrams, format_scored_ngram, max_lcp_value,
        score_dice, score_frequency, score_length, score_mi2, sort_scored_ngrams_for_output,
        source_tags_for_ngram, tokenize_tsv_reader,
    };
    use clap::Parser;
    use std::io::Cursor;

    #[test]
    fn tokenizer_trims_and_lowercases_words() {
        let input = Cursor::new("src\tHuh? don't!\n");
        let corpus = tokenize_tsv_reader(input).unwrap();

        assert_eq!(corpus.token_decoder, vec!["<EOL>", "huh", "don't"]);
        assert_eq!(corpus.tokens, vec![1, 2, 0]);
        assert_eq!(corpus.token_occurrences, vec![1, 1, 1]);
    }

    #[test]
    fn tokenizer_preserves_empty_sentences_as_eol_tokens() {
        let input = Cursor::new("src\tAlpha\nsrc\t\nsrc\tbeta\n");
        let corpus = tokenize_tsv_reader(input).unwrap();

        assert_eq!(corpus.token_decoder, vec!["<EOL>", "alpha", "beta"]);
        assert_eq!(corpus.tokens, vec![1, 0, 0, 2, 0]);
        assert_eq!(corpus.token_occurrences, vec![3, 1, 1]);
    }

    #[test]
    fn tokenizer_keeps_literal_eol_text_distinct_from_eol_tokens() {
        let input = Cursor::new("src\t<EOL> <EOL>\n");
        let corpus = tokenize_tsv_reader(input).unwrap();

        assert_eq!(corpus.token_decoder, vec!["<EOL>", "eol"]);
        assert_eq!(corpus.tokens, vec![1, 1, 0]);
        assert_eq!(corpus.token_occurrences, vec![1, 2]);
    }

    #[test]
    fn tokenizer_ignores_source_tag_words() {
        let input = Cursor::new("tagword\tSentence only\n");
        let corpus = tokenize_tsv_reader(input).unwrap();

        assert_eq!(corpus.token_decoder, vec!["<EOL>", "sentence", "only"]);
        assert_eq!(corpus.tokens, vec![1, 2, 0]);
        assert_eq!(corpus.token_occurrences, vec![1, 1, 1]);
    }

    #[test]
    fn tokenizer_records_zero_based_source_spans_for_tag_changes() {
        let input = Cursor::new("a\tone two\na\tthree\nb\tfour\nb\tfive six\na\tseven\n");
        let corpus = tokenize_tsv_reader(input).unwrap();

        assert_eq!(corpus.sources.len(), 3);
        assert_eq!(corpus.sources[0].source_id, 0);
        assert_eq!(corpus.sources[0].tag, "a");
        assert_eq!(corpus.sources[0].token_start, 0);
        assert_eq!(corpus.sources[0].token_end, 5);
        assert_eq!(corpus.sources[1].source_id, 1);
        assert_eq!(corpus.sources[1].tag, "b");
        assert_eq!(corpus.sources[1].token_start, 5);
        assert_eq!(corpus.sources[1].token_end, 10);
        assert_eq!(corpus.sources[2].source_id, 2);
        assert_eq!(corpus.sources[2].tag, "a");
        assert_eq!(corpus.sources[2].token_start, 10);
        assert_eq!(corpus.sources[2].token_end, 12);
    }

    #[test]
    fn tokenizer_rejects_missing_tab_with_line_number() {
        let input = Cursor::new("src\tok\nbad row\n");
        let error = tokenize_tsv_reader(input).unwrap_err();

        assert_eq!(error, "Malformed TSV row 2: missing tab separator");
    }

    #[test]
    fn tokenizer_rejects_empty_source_tag_with_line_number() {
        let input = Cursor::new("src\tok\n\tbad\n");
        let error = tokenize_tsv_reader(input).unwrap_err();

        assert_eq!(error, "Malformed TSV row 2: empty source tag");
    }

    #[test]
    fn max_lcp_value_reports_largest_entry() {
        assert_eq!(max_lcp_value(&[]), 0);
        assert_eq!(max_lcp_value(&[0, 3, 1, 7, 2]), 7);
    }

    #[test]
    fn output_sort_prefers_score_then_length_then_source_position() {
        let mut scored_ngrams = vec![
            ScoredNgram::new(100.0, 10, 0, 2),
            ScoredNgram::new(100.0, 20, 1, 5),
            ScoredNgram::new(100.0, 15, 2, 5),
            ScoredNgram::new(200.0, 40, 3, 2),
        ];

        sort_scored_ngrams_for_output(&mut scored_ngrams);

        let scores_lengths_and_starts: Vec<(f64, u32, u32)> = scored_ngrams
            .iter()
            .map(|scored_ngram| {
                (
                    scored_ngram.score.0,
                    scored_ngram.len,
                    scored_ngram.token_start,
                )
            })
            .collect();
        assert_eq!(
            scores_lengths_and_starts,
            vec![
                (200.0, 2, 40),
                (100.0, 5, 15),
                (100.0, 5, 20),
                (100.0, 2, 10)
            ]
        );
    }

    #[test]
    fn source_tags_for_ngram_limits_unique_sources() {
        let scored_ngram = ScoredNgram::with_interval(2.0, 0, 3, 2, 0, 3);
        let sarray = vec![0, 3, 6, 9];
        let sources = vec![
            SourceSpan {
                source_id: 0,
                tag: "src-a".to_string(),
                token_start: 0,
                token_end: 6,
            },
            SourceSpan {
                source_id: 1,
                tag: "src-b".to_string(),
                token_start: 6,
                token_end: 9,
            },
            SourceSpan {
                source_id: 2,
                tag: "src-c".to_string(),
                token_start: 9,
                token_end: 12,
            },
        ];

        assert_eq!(
            source_tags_for_ngram(&scored_ngram, &sarray, &sources, 2),
            vec!["src-a", "src-b"]
        );
    }

    #[test]
    fn format_scored_ngram_omits_source_tags_by_default() {
        let scored_ngram = ScoredNgram::with_interval(2.0, 0, 0, 2, 0, 0);
        let tokens = vec![1, 2, 0];
        let token_decoder = vec!["<EOL>".to_string(), "a".to_string(), "b".to_string()];
        let sarray = vec![0];
        let sources = vec![SourceSpan {
            source_id: 0,
            tag: "src-a".to_string(),
            token_start: 0,
            token_end: 3,
        }];

        assert_eq!(
            format_scored_ngram(&scored_ngram, &tokens, &token_decoder, &sarray, &sources, 0),
            "a b (2)"
        );
    }

    #[test]
    fn format_scored_ngram_appends_source_tags_when_requested() {
        let scored_ngram = ScoredNgram::with_interval(2.0, 0, 0, 2, 0, 0);
        let tokens = vec![1, 2, 0];
        let token_decoder = vec!["<EOL>".to_string(), "a".to_string(), "b".to_string()];
        let sarray = vec![0];
        let sources = vec![SourceSpan {
            source_id: 0,
            tag: "src-a".to_string(),
            token_start: 0,
            token_end: 3,
        }];

        assert_eq!(
            format_scored_ngram(&scored_ngram, &tokens, &token_decoder, &sarray, &sources, 1),
            "a b (2)\t[src-a]"
        );
    }

    #[test]
    fn args_parse_rank_and_size_bounds() {
        let args = Args::try_parse_from([
            "ngrams",
            "--rank-by",
            "length",
            "--min-ngram-size",
            "3",
            "--max-ngram-size",
            "7",
            "--max-heap-size",
            "123",
            "--min-count",
            "4",
            "--max-source-tags",
            "2",
            "input.txt",
        ])
        .unwrap();

        assert_eq!(args.rank_by, RankBy::Length);
        assert_eq!(args.min_ngram_size, 3);
        assert_eq!(args.max_ngram_size, 7);
        assert_eq!(args.max_heap_size, 123);
        assert_eq!(args.min_count, 4);
        assert_eq!(args.max_source_tags, 2);
        assert!(args.validate().is_ok());
    }

    #[test]
    fn args_reject_zero_min_ngram_size() {
        let args = Args::try_parse_from(["ngrams", "-n", "0", "input.txt"]).unwrap();

        assert_eq!(
            args.validate().unwrap_err(),
            "--min-ngram-size must be at least 1"
        );
    }

    #[test]
    fn args_reject_max_ngram_size_below_min_ngram_size() {
        let args = Args::try_parse_from(["ngrams", "-n", "4", "-x", "3", "input.txt"]).unwrap();

        assert_eq!(
            args.validate().unwrap_err(),
            "--max-ngram-size (3) must be greater than or equal to --min-ngram-size (4)"
        );
    }

    #[test]
    fn args_accept_zero_max_heap_size_as_unbounded() {
        let args = Args::try_parse_from(["ngrams", "-k", "0", "input.txt"]).unwrap();

        assert_eq!(args.max_heap_size, 0);
        assert!(args.validate().is_ok());
    }

    #[test]
    fn zero_sized_heap_limit_keeps_every_scored_ngram() {
        let mut heap = ScoredNgramHeap::new(0);

        heap.push(ScoredNgram::new(1.0, 10, 0, 2));
        heap.push(ScoredNgram::new(2.0, 20, 1, 2));
        heap.push(ScoredNgram::new(3.0, 30, 2, 2));

        assert_eq!(heap.into_vec().len(), 3);
    }

    #[test]
    fn args_reject_min_count_below_repeated_threshold() {
        let zero_count_args = Args::try_parse_from(["ngrams", "-c", "0", "input.txt"]).unwrap();
        let one_count_args = Args::try_parse_from(["ngrams", "-c", "1", "input.txt"]).unwrap();

        assert_eq!(
            zero_count_args.validate().unwrap_err(),
            "--min-count must be at least 2 because enumeration is repeated-only"
        );
        assert_eq!(
            one_count_args.validate().unwrap_err(),
            "--min-count must be at least 2 because enumeration is repeated-only"
        );
    }

    #[test]
    fn scoring_functions_use_the_same_direct_arguments() {
        let ngram_tokens = vec![1, 2];
        let ngram_occurrences = 2;
        let token_occurrences = vec![0, 10, 5];
        let corpus_token_count = 100;
        let pmi = (2.0_f64 * 100.0_f64 / (10.0_f64 * 5.0_f64)).ln();
        let mi2 = score_mi2(
            &ngram_tokens,
            ngram_occurrences,
            &token_occurrences,
            corpus_token_count,
        )
        .unwrap();

        assert!((mi2 - (ngram_occurrences as f64 * pmi * pmi)).abs() < f64::EPSILON * 8.0);
        assert_eq!(
            score_dice(
                &ngram_tokens,
                ngram_occurrences,
                &token_occurrences,
                corpus_token_count
            ),
            Some(2.0 * 2.0 / 15.0)
        );
        assert_eq!(
            score_frequency(
                &ngram_tokens,
                ngram_occurrences,
                &token_occurrences,
                corpus_token_count
            ),
            Some(2.0)
        );
        assert_eq!(
            score_length(
                &ngram_tokens,
                ngram_occurrences,
                &token_occurrences,
                corpus_token_count
            ),
            Some(2.0)
        );
    }

    #[test]
    fn frequency_score_is_not_gated_by_positive_pmi() {
        let ngram_tokens = vec![1, 2];
        let ngram_occurrences = 10;
        let token_occurrences = vec![0, 100, 100];
        let corpus_token_count = 100;

        assert_eq!(
            score_mi2(
                &ngram_tokens,
                ngram_occurrences,
                &token_occurrences,
                corpus_token_count
            ),
            None
        );
        assert_eq!(
            score_frequency(
                &ngram_tokens,
                ngram_occurrences,
                &token_occurrences,
                corpus_token_count
            ),
            Some(10.0)
        );
    }

    #[test]
    fn selected_frequency_ranking_keeps_low_pmi_candidates() {
        let mut corpus = String::new();
        for _ in 0..10 {
            corpus.push_str("src\ta b\n");
        }
        for _ in 0..90 {
            corpus.push_str("src\ta c\n");
        }
        for _ in 0..90 {
            corpus.push_str("src\td b\n");
        }

        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            2,
            2000,
            10,
            RankBy::Frequency,
        );

        assert!(scored_ngrams.iter().any(|scored_ngram| {
            scored_ngram.score.0 == 10.0
                && corpus.tokens
                    [scored_ngram.token_start as usize..scored_ngram.token_start as usize + 2]
                    == [1, 2]
        }));
    }

    #[test]
    fn frequency_ranking_emits_only_longest_owned_length_per_interval() {
        let corpus = "src\ta b c\nsrc\ta b c\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            3,
            0,
            2,
            RankBy::Frequency,
        );

        let emitted: Vec<Vec<_>> = scored_ngrams
            .iter()
            .map(|scored_ngram| {
                let token_start = scored_ngram.token_start as usize;
                let token_end = token_start + scored_ngram.len as usize;
                corpus.tokens[token_start..token_end].to_vec()
            })
            .collect();

        assert!(emitted.contains(&vec![1, 2, 3]));
        assert!(emitted.contains(&vec![2, 3]));
        assert!(!emitted.contains(&vec![1, 2]));
    }

    #[test]
    fn mi2_ranking_emits_only_longest_owned_length_per_interval() {
        let corpus = "src\ta b c\nsrc\ta b c\nsrc\td e f\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            3,
            0,
            2,
            RankBy::Mi2,
        );

        let emitted: Vec<Vec<_>> = scored_ngrams
            .iter()
            .map(|scored_ngram| {
                let token_start = scored_ngram.token_start as usize;
                let token_end = token_start + scored_ngram.len as usize;
                corpus.tokens[token_start..token_end].to_vec()
            })
            .collect();

        assert!(emitted.contains(&vec![1, 2, 3]));
        assert!(emitted.contains(&vec![2, 3]));
        assert!(!emitted.contains(&vec![1, 2]));
    }

    #[test]
    fn length_ranking_finds_longest_repeated_ngram() {
        let corpus = "src\ta b c\nsrc\tx y\nsrc\ta b c\nsrc\tx y\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            1,
            3,
            1,
            2,
            RankBy::Length,
        );

        let longest = scored_ngrams.first().unwrap();
        let token_start = longest.token_start as usize;
        let token_end = token_start + longest.len as usize;

        assert_eq!(longest.score.0, 3.0);
        assert_eq!(corpus.tokens[token_start..token_end], [1, 2, 3]);
    }

    #[test]
    fn stack_enumeration_scores_repeated_ngram_once() {
        let corpus = "src\ta b\nsrc\tx a b\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            2,
            2000,
            2,
            RankBy::Frequency,
        );

        let repeated_bigrams: Vec<&ScoredNgram> = scored_ngrams
            .iter()
            .filter(|scored_ngram| {
                let token_start = scored_ngram.token_start as usize;
                scored_ngram.score.0 == 2.0 && corpus.tokens[token_start..token_start + 2] == [1, 2]
            })
            .collect();

        assert_eq!(repeated_bigrams.len(), 1);
    }

    #[test]
    fn stack_enumeration_keeps_suffix_interval_bounds() {
        let corpus = "src\ta b\nsrc\tx a b\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            2,
            2000,
            2,
            RankBy::Frequency,
        );

        let repeated_bigram = scored_ngrams
            .iter()
            .find(|scored_ngram| {
                let token_start = scored_ngram.token_start as usize;
                corpus.tokens[token_start..token_start + 2] == [1, 2]
            })
            .unwrap();
        let left = repeated_bigram.left as usize;
        let right = repeated_bigram.right as usize;
        let mut occurrence_starts: Vec<usize> = sarray[left..=right]
            .iter()
            .map(|&token_start| token_start as usize)
            .collect();
        occurrence_starts.sort_unstable();

        assert_eq!(right - left + 1, 2);
        assert_eq!(occurrence_starts, vec![0, 4]);
    }

    #[test]
    fn stack_enumeration_skips_singletons() {
        let corpus = "src\ta b\nsrc\tc d\n";
        let corpus = tokenize_tsv_reader(Cursor::new(corpus)).unwrap();
        let sarray = build_suffix_array(&corpus.tokens);
        let lcp = build_lcp_array(&sarray, &corpus.tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &corpus.tokens,
            &corpus.token_occurrences,
            2,
            2,
            2000,
            2,
            RankBy::Frequency,
        );

        assert!(scored_ngrams.is_empty());
    }
}
