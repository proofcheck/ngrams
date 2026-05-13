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
const MIN_POSITIVE_PMI: f64 = 0.1;

/// Convert each unique word to a unique token value.
///
/// End-of-line (`<EOL>`) is assigned token value 0.
/// Returns:
/// - A vector of tokens
/// - A decoder vector mapping tokens back to words
/// - A token-occurrence vector where token_occurrences[i] is the number of occurrences of token i
fn tokenize_reader<R: BufRead>(mut reader: R) -> (Vec<Token>, Vec<String>, Vec<usize>) {
    eprintln!("Tokenizing file");

    let mut token_map = HashMap::new();
    let mut token_decoder = vec!["<EOL>".to_string()];
    let mut token_occurrences = vec![0usize];
    let mut token_vec = Vec::new();
    let mut line = String::new();
    let mut line_number = 0usize;

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).unwrap();
        if bytes_read == 0 {
            break;
        }

        for word_with_punct in line.split_whitespace() {
            // Check the raw token first so already-normalized entries avoid the
            // trimming and lowercasing work entirely.
            let token = if let Some(&token) = token_map.get(word_with_punct) {
                token
            } else {
                let trimmed_word = word_with_punct
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_ascii_control());

                if trimmed_word.is_empty() {
                    continue;
                }

                let normalized_word = trimmed_word.to_lowercase();
                if let Some(&token) = token_map.get(normalized_word.as_str()) {
                    token
                } else {
                    if token_decoder.len() == Token::MAX as usize {
                        panic!("Too many unique tokens (max: {})", Token::MAX);
                    }
                    let new_token = token_decoder.len() as Token;
                    token_map.insert(normalized_word.clone(), new_token);
                    token_decoder.push(normalized_word);
                    token_occurrences.push(0);
                    new_token
                }
            };

            token_vec.push(token);
            token_occurrences[token as usize] += 1;
        }

        token_vec.push(0);
        token_occurrences[0] += 1;

        if line_number % 1000000 == 0 {
            eprintln!("Processed {} lines", line_number);
        }
        line_number += 1;
    }

    eprintln!(
        "Processed {} tokens, {} distinct",
        token_vec.len(),
        token_decoder.len()
    );
    eprintln!("Counted {} tokens", token_occurrences.len());

    (token_vec, token_decoder, token_occurrences)
}

/// Convert each unique word in a file to a unique token value.
fn tokenize_file(path: &str) -> (Vec<Token>, Vec<String>, Vec<usize>) {
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);
    tokenize_reader(reader)
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
}

/// Command line arguments for the n-gram ranking tool.
#[derive(Parser)]
#[command(about = "Text search tool using suffix arrays", version)]
struct Args {
    /// Text file to search in
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

    /// Minimum number of times an n-gram must occur to be scored
    #[arg(short = 'c', long, default_value_t = DEFAULT_MIN_COUNT)]
    min_count: usize,
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
        if self.min_count == 0 {
            return Err("--min-count must be at least 1".to_string());
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
/// the suffixes at sarray[i] and sarray[i+1]
fn build_lcp_array(sarray: &[Index], tokens: &[Token]) -> Vec<u32> {
    eprintln!("Building LCP array");

    // Initialize the LCP array with zeros
    let mut lcp: Vec<u32> = vec![0; sarray.len()];

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
    let mut h: u32 = 0;

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
        while i + (h as usize) < sarray.len()
            && j + (h as usize) < sarray.len()
            && tokens[i + (h as usize)] == tokens[j + (h as usize)]
            && tokens[i + (h as usize)] != 0
            && tokens[j + (h as usize)] != 0
        {
            h += 1;
        }

        // Store the LCP value at the position of suffix i in the sorted array
        lcp[inv_sarray[i] as usize] = h;

        // Decrease h by 1 for the next iteration (Kasai's optimization)
        // saturating_sub ensures we don't go below 0
        h = h.saturating_sub(1);
    }

    eprintln!("Built LCP array");
    // eprintln!("LCP array: {:?}", lcp);
    lcp
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
}

impl ScoredNgram {
    fn new(score: f64, token_start: Index, suffix_index: Index, len: Index) -> ScoredNgram {
        ScoredNgram {
            score: OrderedFloat(score),
            token_start,
            suffix_index,
            len,
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

/// Count how many suffixes share the candidate prefix at a given suffix-array position.
fn count_ngram_occurrences(suffix_index: usize, len: usize, lcp: &[u32]) -> usize {
    // Expand left and right across the contiguous suffix block with this prefix.
    let mut left = suffix_index;
    let mut right = suffix_index;
    while left > 0 && lcp[left - 1] as usize >= len {
        left -= 1;
    }

    while right < lcp.len() && lcp[right] as usize >= len {
        right += 1;
    }

    right - left + 1
}

/// Enumerate unique n-gram candidates, score them, and retain the best matches.
///
/// This function uses the suffix array and LCP array to efficiently enumerate all unique
/// substrings without explicitly storing them in a hash set. Each suffix contributes new
/// unique substrings that are longer than its LCP with the next suffix in suffix-array order.
fn collect_top_scored_ngrams(
    sarray: &[Index],
    lcp: &[u32],
    tokens: &[Token],
    token_occurrences: &[usize],
    min_ngram_size: usize,
    max_ngram_size: usize,
    max_heap_size: usize,
    min_count: usize,
    scoring_function: ScoringFunction,
) -> Vec<ScoredNgram> {
    // Key insight: As substring length increases, the valid matching range [left, right]
    // can only shrink. We incrementally refine boundaries as we process longer substrings.
    eprintln!("Scoring ngrams");

    let corpus_token_count = token_occurrences.iter().sum::<usize>();
    let mut best_ngrams = ScoredNgramHeap::new(max_heap_size);

    for suffix_index in 0..sarray.len() {
        let suffix_start = sarray[suffix_index] as usize;
        let suffix_len = (tokens.len() - suffix_start).min(max_ngram_size);

        // Skip substrings that will be emitted by the suffix immediately to the left in the
        // shared-prefix block. With lcp[suffix_index] = LCP(sarray[i], sarray[i + 1]),
        // this suffix only owns substrings longer than that LCP.
        let min_unique_len = lcp[suffix_index] as usize + 1;

        for len in min_unique_len..=suffix_len {
            // Stop if we encounter an EOL token (i.e., don't cross line boundaries)
            if tokens[suffix_start + len - 1] == 0 {
                break;
            }

            // Respect the caller's lower n-gram bound after preserving EOL detection.
            if len < min_ngram_size {
                continue;
            }

            let ngram_tokens = &tokens[suffix_start..suffix_start + len];
            let ngram_occurrences = count_ngram_occurrences(suffix_index, len, lcp);

            if ngram_occurrences < min_count {
                continue;
            }

            if let Some(score) = scoring_function(
                ngram_tokens,
                ngram_occurrences,
                token_occurrences,
                corpus_token_count,
            ) {
                best_ngrams.push(ScoredNgram::new(
                    score,
                    suffix_start as Index,
                    suffix_index as Index,
                    len as Index,
                ));
            }
        }
    }

    let mut scored_ngrams = best_ngrams.into_vec();
    sort_scored_ngrams_for_output(&mut scored_ngrams);
    scored_ngrams
}

/// Print scored n-grams in their already-ranked order.
fn print_scored_ngrams(scored_ngrams: &[ScoredNgram], tokens: &[Token], token_decoder: &[String]) {
    for scored_ngram in scored_ngrams {
        let suffix_start = scored_ngram.token_start as usize;
        let suffix_len = scored_ngram.len as usize;
        let ngram_tokens = &tokens[suffix_start..suffix_start + suffix_len];
        let words: Vec<&str> = ngram_tokens
            .iter()
            .map(|&t| token_decoder[t as usize].as_str())
            .collect();
        println!("{} ({})", words.join(" "), scored_ngram.score.0);
    }
}

fn main() {
    let args = Args::parse_validated();
    let path = &args.file;

    let (tokens, token_decoder, token_occurrences) = tokenize_file(path);
    // eprintln!("{:?} tokens", tokens);
    // eprintln!("{:?} token_decoder", token_decoder);
    // eprintln!("Token occurrences: {:?}", token_occurrences);

    let sarray = build_suffix_array(&tokens);

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

    let lcp = build_lcp_array(&sarray, &tokens);

    // for i in 0..sarray.len() {
    //     println!("{} {} {} {}", sarray[i], tokens[sarray[i] as usize],token_decoder[tokens[sarray[i] as usize] as usize], lcp[i]);
    // }
    let scoring_function = scoring_function_for(args.rank_by);
    let scored_ngrams = collect_top_scored_ngrams(
        &sarray,
        &lcp,
        &tokens,
        &token_occurrences,
        args.min_ngram_size,
        args.max_ngram_size,
        args.max_heap_size,
        args.min_count,
        scoring_function,
    );
    print_scored_ngrams(&scored_ngrams, &tokens, &token_decoder);
}

#[cfg(test)]
mod tests {
    use super::{
        Args, RankBy, ScoredNgram, ScoredNgramHeap, build_lcp_array, build_suffix_array,
        collect_top_scored_ngrams, score_dice, score_frequency, score_mi2, scoring_function_for,
        sort_scored_ngrams_for_output, tokenize_reader,
    };
    use clap::Parser;
    use std::io::Cursor;

    #[test]
    fn tokenizer_trims_and_lowercases_words() {
        let input = Cursor::new("Huh? don't!\n");
        let (tokens, decoder, token_occurrences) = tokenize_reader(input);

        assert_eq!(decoder, vec!["<EOL>", "huh", "don't"]);
        assert_eq!(tokens, vec![1, 2, 0]);
        assert_eq!(token_occurrences, vec![1, 1, 1]);
    }

    #[test]
    fn tokenizer_preserves_empty_lines_as_eol_tokens() {
        let input = Cursor::new("Alpha\n\nbeta\n");
        let (tokens, decoder, token_occurrences) = tokenize_reader(input);

        assert_eq!(decoder, vec!["<EOL>", "alpha", "beta"]);
        assert_eq!(tokens, vec![1, 0, 0, 2, 0]);
        assert_eq!(token_occurrences, vec![3, 1, 1]);
    }

    #[test]
    fn tokenizer_keeps_literal_eol_text_distinct_from_eol_tokens() {
        let input = Cursor::new("<EOL> <EOL>\n");
        let (tokens, decoder, token_occurrences) = tokenize_reader(input);

        assert_eq!(decoder, vec!["<EOL>", "eol"]);
        assert_eq!(tokens, vec![1, 1, 0]);
        assert_eq!(token_occurrences, vec![1, 2]);
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
    fn args_parse_rank_and_size_bounds() {
        let args = Args::try_parse_from([
            "ngrams",
            "--rank-by",
            "dice",
            "--min-ngram-size",
            "3",
            "--max-ngram-size",
            "7",
            "--max-heap-size",
            "123",
            "--min-count",
            "4",
            "input.txt",
        ])
        .unwrap();

        assert_eq!(args.rank_by, RankBy::Dice);
        assert_eq!(args.min_ngram_size, 3);
        assert_eq!(args.max_ngram_size, 7);
        assert_eq!(args.max_heap_size, 123);
        assert_eq!(args.min_count, 4);
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
    fn args_reject_zero_min_count() {
        let args = Args::try_parse_from(["ngrams", "-c", "0", "input.txt"]).unwrap();

        assert_eq!(
            args.validate().unwrap_err(),
            "--min-count must be at least 1"
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
            corpus.push_str("a b\n");
        }
        for _ in 0..90 {
            corpus.push_str("a c\n");
        }
        for _ in 0..90 {
            corpus.push_str("d b\n");
        }

        let (tokens, _decoder, token_occurrences) = tokenize_reader(Cursor::new(corpus));
        let sarray = build_suffix_array(&tokens);
        let lcp = build_lcp_array(&sarray, &tokens);
        let scored_ngrams = collect_top_scored_ngrams(
            &sarray,
            &lcp,
            &tokens,
            &token_occurrences,
            2,
            2,
            2000,
            10,
            scoring_function_for(RankBy::Frequency),
        );

        assert!(scored_ngrams.iter().any(|scored_ngram| {
            scored_ngram.score.0 == 10.0
                && tokens[scored_ngram.token_start as usize..scored_ngram.token_start as usize + 2]
                    == [1, 2]
        }));
    }
}
