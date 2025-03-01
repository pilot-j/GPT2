import regex
from collections import defaultdict
from typing import List, Dict, Tuple, Any


class BytePairEncoder:
    """
    A class implementing Byte Pair Encoding (BPE) compression algorithm.
    
    BPE iteratively replaces the most frequent pair of consecutive bytes/tokens
    with a new token, effectively compressing the data by reducing repeated patterns.
    """
    
    def __init__(self, lower_frequency_limit: int = 1, max_iterations: int = 100, vocabulary:Dict[Tuple[Any, Any], int] = {} ):
        """
        Initialize the BPE encoder.
        
        Args:
            lower_frequency_limit: Minimum frequency threshold for pair merging
            max_iterations: Maximum number of compression iterations
            vocabulary: predefined dictionary of current vocabulary in use
        """
        self.lower_frequency_limit = lower_frequency_limit
        self.max_iterations = max_iterations
        self.vocab = vocabulary
        
    def _count_pair_frequencies(self, sequence: List[int]) -> Dict[Tuple[Any, Any], int]:
        """
        Count frequencies of adjacent pairs in the sequence.
        
        Args:
            sequence: List of tokens to analyze
            
        Returns:
            Dictionary mapping token pairs to their frequencies
        """
        freq_map = defaultdict(int)
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            freq_map[pair] += 1
        return freq_map
    
    def _find_most_frequent_pairs(self, freq_map: Dict[Tuple[Any, Any], int]) -> List[Tuple[Any, Any]]:
        """
        Find pairs that occur most frequently, above the lower frequency limit.
        
        Args:
            freq_map: Dictionary of pair frequencies
            
        Returns:
            List of pairs that meet the frequency criteria
        """
        if not freq_map:
            return []
        most_freq_pairs = [x for x in freq_map.keys() if freq_map[x] == max(max(freq_map.values()), self.lower_frequency_limit)]
        return most_freq_pairs

    

    
    def _update_vocabulary(self, pairs):
        """
        Assign new token values to frequent pairs in the vocabulary.
        
        Args:
            pairs: List of pairs to add to vocabulary
        """
        if not self.vocab:
            next_token = 0
        else:
            next_token = max(self.vocab.values()) + 1
            
        for pair in pairs:
            self.vocab[pair] = next_token
            next_token += 1
    
    def _compress_sequence(self, sequence: List[Any], vocab) -> List[Any]:
        """
        Compress the sequence by replacing frequent pairs with their token values.
        
        Args:
            sequence: List of tokens to compress
            vocabulary
            
        Returns:
            Compressed sequence
        """
        compressed = []
        i, flag = 0, 0
        while i < len(sequence) - 1:
            pair = (sequence[i], sequence[i + 1])
            if pair in vocab:
                flag = 1
                compressed.append(vocab[pair])
                i += 2
            else:
                compressed.append(sequence[i])
                i += 1
                
        # Handle last token if we're at the end
        if i == len(sequence) - 1:
            compressed.append(sequence[-1])

        return compressed

    def regex_chunking(self, text: str) -> List[str]:
        pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        text_chunks = regex.findall(pattern , text)
        return text_chunks
    
    def train_bpe(self, train_text: str):
        """
        Encode the input sequence using BPE algorithm.
        
        Args:
            sequence: Input sequence to compress
            
        Returns:
            None
        """
        text_chunks = self.regex_chunking(train_text)
        
        current_sequence_list = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        
        
        
        for i in range(self.max_iterations):
            freq_map = defaultdict(int)
            
            # Count frequencies of adjacent pairs
            for seq in current_sequence_list:
                for k, v in self._count_pair_frequencies(seq).items():
                    freq_map[k] += v

            #Find frequent pairs. Note multiple pairs can have same frequency.
            frequent_pairs = self._find_most_frequent_pairs(freq_map)
            
           
            # If no pairs meet criteria, stop iteration
            if not frequent_pairs:
                print("No more frequent pairs found. Training completed!")
                break
                
            # Update vocabulary with new tokens
            self._update_vocabulary(frequent_pairs)

          
            # Compress sequence using updated vocabulary
            new_sequence_list = [self._compress_sequence(seq, self.vocab) for seq in current_sequence_list]   
            current_sequence_list = new_sequence_list

        print(f"length of training corpus {len(train_text)}, #new_tokens added:{len(self.vocab)-256}. Training completed")
        

    def encode(self, text):
        pass
        
            
    def get_vocabulary(self) -> Dict[Tuple[Any, Any], int]:
        """
        Get the current BPE vocabulary.
        
        Returns:
            Dictionary mapping token pairs to their assigned values
        """
        return self.vocab.copy()

    

    
