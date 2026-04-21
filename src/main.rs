use clap::Parser;
use ordered_float::OrderedFloat;
use rayon::slice::ParallelSliceMut;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{self, BufRead};

type Token = u32; // u16;
type Index = u32;

/// Convert each unique word to a unique token value.
///
/// End-of-line (`<EOL>`) is assigned token value 0.
/// Returns:
/// - A vector of tokens
/// - A decoder vector mapping tokens back to words
/// - A counts vector where counts[i] is the number of occurrences of token i
fn tokenize_reader<R: BufRead>(mut reader: R) -> (Vec<Token>, Vec<String>, Vec<usize>) {
    eprintln!("Tokenizing file");

    let mut token_map = HashMap::new();
    token_map.insert("<EOL>".to_string(), 0);
    let mut token_decoder = vec!["<EOL>".to_string()];
    let mut counts = vec![0usize];
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
            let trimmed_word = word_with_punct
                .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_ascii_control());

            if trimmed_word.is_empty() {
                continue;
            }

            let normalized_word = trimmed_word.to_lowercase();
            let token = if let Some(&token) = token_map.get(normalized_word.as_str()) {
                token
            } else {
                if token_decoder.len() == Token::MAX as usize {
                    panic!("Too many unique tokens (max: {})", Token::MAX);
                }
                let new_token = token_decoder.len() as Token;
                token_map.insert(normalized_word.clone(), new_token);
                token_decoder.push(normalized_word);
                counts.push(0);
                new_token
            };

            token_vec.push(token);
            counts[token as usize] += 1;
        }

        token_vec.push(0);
        counts[0] += 1;

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
    eprintln!("Counted {} tokens", counts.len());

    (token_vec, token_decoder, counts)
}

/// Convert each unique word in a file to a unique token value.
fn tokenize_file(path: &str) -> (Vec<Token>, Vec<String>, Vec<usize>) {
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);
    tokenize_reader(reader)
}

/// Command line arguments for the sarray tool
#[derive(Parser)]
#[command(about = "Text search tool using suffix arrays", version)]
struct Args {
    /// Text file to search in
    #[arg(required = true)]
    file: String,
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
fn build_lcp_array(sarray: &[Index], tokens: &[Token]) -> Vec<Index> {
    eprintln!("Building LCP array");

    // Initialize the LCP array with zeros
    let mut lcp: Vec<Index> = vec![0; sarray.len()];

    // Build the inverse suffix array: inv_sarray[i] tells us the position in sarray
    // where suffix i appears. This allows us to quickly find a suffix's position
    // in the sorted order.
    let mut inv_sarray: Vec<usize> = vec![0; sarray.len()];
    for (i, &s) in sarray.iter().enumerate() {
        inv_sarray[s as usize] = i;
    }

    // h tracks the LCP length from the previous iteration
    // Kasai's algorithm exploits the fact that if suffix i has LCP h with its successor,
    // then suffix i+1 has LCP at least h-1 with its successor (this is the key insight
    // that makes the algorithm O(n) instead of O(n²))
    let mut h = 0;

    // Iterate through suffixes in text order (not sorted order)
    for i in 0..sarray.len() {
        // If this suffix is the last one in sorted order, it has no successor to compare with
        if inv_sarray[i] == sarray.len() - 1 {
            h = 0;
            continue;
        }

        // Find the next suffix in sorted order after suffix i
        let j = sarray[inv_sarray[i] + 1] as usize;

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
        lcp[inv_sarray[i]] = h as Index;

        // Decrease h by 1 for the next iteration (Kasai's optimization)
        // saturating_sub ensures we don't go below 0
        h = h.saturating_sub(1);
    }

    eprintln!("Built LCP array");
    // eprintln!("LCP array: {:?}", lcp);
    lcp
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct InterestingRange {
    score: OrderedFloat<f64>,
    start: Index,
    end: Index,
}

impl InterestingRange {
    fn new(score: f64, start: Index, end: Index) -> InterestingRange {
        InterestingRange {
            score: OrderedFloat(score),
            start,
            end,
        }
    }
}

// Implement a *min* heap of InterestingRange
struct InterestingRangeHeap {
    max_size: usize,
    heap: BinaryHeap<Reverse<InterestingRange>>,
}

impl InterestingRangeHeap {
    fn new(max_size: usize) -> InterestingRangeHeap {
        InterestingRangeHeap {
            max_size,
            heap: BinaryHeap::new(),
        }
    }
    fn push(&mut self, range: InterestingRange) {
        // Add the new range if the heap is not full
        // and replace the heap's minimum by the new range if the
        // new range is bigger.
        if self.heap.len() < self.max_size {
            self.heap.push(Reverse(range));
        } else if range > self.heap.peek().unwrap().0 {
            self.heap.pop();
            self.heap.push(Reverse(range));
        }
    }
}

// debugging output for an InterestingRangeHeap
impl std::fmt::Debug for InterestingRangeHeap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterestingRangeHeap")
            .field("max_size", &self.max_size)
            .field("heap", &self.heap)
            .finish()
    }
}

/// Print all unique substrings from the token vector along with their occurrence counts
///
/// This function uses the suffix array and LCP array to efficiently enumerate all unique
/// substrings without explicitly storing them in a hash set. The key insights are:
/// - Each suffix contributes new unique substrings that are longer than its LCP with the previous suffix
/// - The number of occurrences can be found by looking at the range of suffixes that share the same prefix
/// - Uses incremental boundary refinement to find occurrence ranges in O(n) amortized time per suffix
///
/// # Arguments
/// * `sarray` - The suffix array (sorted indices)
/// * `lcp` - The LCP array (longest common prefix lengths)
/// * `tokens` - The original token vector
/// * `token_decoder` - Maps token IDs back to their string representations
fn print_unique_substrings(
    sarray: &[Index],
    lcp: &[Index],
    tokens: &[Token],
    token_decoder: &[String],
    counts: &[usize],
) {
    // Key insight: As substring length increases, the valid matching range [left, right]
    // can only shrink. We incrementally refine boundaries as we process longer substrings.
    eprintln!("Scoring ngrams");

    const MAX_NGRAM_LEN: usize = 15;

    let total_count = counts.iter().sum::<usize>();

    let mut best_mi2 = InterestingRangeHeap::new(2000);
    let mut best_dice = InterestingRangeHeap::new(2000);

    const DEBUG: bool = false;

    for i in 0..sarray.len() {
        // eprintln!("Suffix {}", i);
        let suffix_start = sarray[i] as usize;
        let suffix_len = (tokens.len() - suffix_start).min(MAX_NGRAM_LEN);

        // Skip substrings already enumerated by the previous suffix (those with length <= lcp[i])
        let min_len = lcp[i] as usize + 1;

        for len in min_len..=suffix_len {
            // Stop if we encounter an EOL token (i.e., don't cross line boundaries)
            if tokens[suffix_start + len - 1] == 0 {
                break;
            }

            // Skip unigrams
            if len < 2 {
                continue;
            }

            let substring_tokens = &tokens[suffix_start..suffix_start + len];

            if DEBUG {
                let words: Vec<&str> = substring_tokens
                    .iter()
                    .map(|&t| token_decoder[t as usize].as_str())
                    .collect();

                println!("{}", words.join(" "));
            }

            // Find all suffixes that share this prefix by expanding boundaries
            let mut left = i;
            let mut right = i;
            while left > 0 && lcp[left - 1] as usize >= len {
                // eprintln!("Left: {}, lcp[left - 1] = {}, len = {}", left, lcp[left - 1], len);
                left -= 1;
            }

            while right < lcp.len() && lcp[right] as usize >= len {
                right += 1;
            }

            let count = right - left + 1;
            // eprintln!("count = {}", count);

            // Calculate PMI (Pointwise Mutual Information) for this n-gram if count>=5

            const MIN_COUNT: usize = 10;

            if count >= MIN_COUNT {
                // the numerator of the fraction is the number of times this n-gram appears
                // times the total number of n-grams
                let numerator = (count as f64) * (total_count as f64).powf((len - 1) as f64);

                // the denominator of the fraction is the product of the number of times each
                // word in the n-gram appears, where the ngram consists of the tokens
                // in substring_tokens.
                let denominator = substring_tokens
                    .iter()
                    .map(|&t| counts[t as usize] as f64)
                    .product::<f64>();

                // the PMI is the natural log of the fraction

                let pmi = numerator.ln() - denominator.ln();

                if pmi > 0.1 {
                    let mi2 = count as f64 * pmi * pmi;
                    let dice = (len as f64) * (count as f64)
                        / (substring_tokens
                            .iter()
                            .map(|&t| counts[t as usize] as f64)
                            .sum::<f64>());
                    if DEBUG {
                        println!("PMI: {}", pmi);
                        println!("MI2: {}", mi2);
                    }
                    best_mi2.push(InterestingRange::new(mi2, i as u32, (i + len) as u32));
                    best_dice.push(InterestingRange::new(dice, i as u32, (i + len) as u32));
                }
            }
        }
    }

    // Iterate throught the best tokens by mi2
    let mut mi2_ranges: Vec<InterestingRange> =
        best_mi2.heap.drain().map(|wrapped| wrapped.0).collect();
    mi2_ranges.sort_by(|a, b| b.score.cmp(&a.score));

    for range in mi2_ranges {
        let suffix_start = sarray[range.start as usize] as usize;
        let suffix_len = range.end - range.start;
        let substring_tokens = &tokens[suffix_start..suffix_start + suffix_len as usize];
        let words: Vec<&str> = substring_tokens
            .iter()
            .map(|&t| token_decoder[t as usize].as_str())
            .collect();
        println!("{} ({})", words.join(" "), range.score.0);
    }
}

fn main() {
    let args = Args::parse();
    let path = &args.file;

    let (tokens, token_decoder, counts) = tokenize_file(path);
    // eprintln!("{:?} tokens", tokens);
    // eprintln!("{:?} token_decoder", token_decoder);
    // eprintln!("Token counts: {:?}", counts);

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
    print_unique_substrings(&sarray, &lcp, &tokens, &token_decoder, &counts);
}

#[cfg(test)]
mod tests {
    use super::tokenize_reader;
    use std::io::Cursor;

    #[test]
    fn tokenizer_trims_and_lowercases_words() {
        let input = Cursor::new("Huh? don't!\n");
        let (tokens, decoder, counts) = tokenize_reader(input);

        assert_eq!(decoder, vec!["<EOL>", "huh", "don't"]);
        assert_eq!(tokens, vec![1, 2, 0]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn tokenizer_preserves_empty_lines_as_eol_tokens() {
        let input = Cursor::new("Alpha\n\nbeta\n");
        let (tokens, decoder, counts) = tokenize_reader(input);

        assert_eq!(decoder, vec!["<EOL>", "alpha", "beta"]);
        assert_eq!(tokens, vec![1, 0, 0, 2, 0]);
        assert_eq!(counts, vec![3, 1, 1]);
    }
}
