import re
from collections import Counter


class DictionarySearcher:
    def __init__(self):
        self.words_map = {}
        self.must_have = []
        self.must_have_count = 0
        self.must_not = []
        self.query_chunk_pattern = re.compile("\s+")

    def add_patterns(self, patterns):
        patterns = self.query_chunk_pattern.split(patterns.strip())
        for word in patterns:
            if word != "":
                self.add_word(word)

    def match(self, sentence):
        results = self.search(sentence, False)
        counter = Counter()
        must_match_indexes = set()
        for item in results:
            word = item["word"]
            for index, group in enumerate(self.must_have):
                if word in group:
                    must_match_indexes.add(index)
                    counter.update([index])
            for index, group in enumerate(self.must_not):
                if word in group:
                    return False, 0
        if len(must_match_indexes) == self.must_have_count:
            return True, counter.most_common()[-1][1]
        return False, 0

    def add_word(self, word):
        word_list = list(word)
        for i in range(len(word_list) - 1):
            middle_word = "".join(word_list[: i + 1])
            if middle_word not in self.words_map:
                self.words_map[middle_word] = False
        self.words_map[word] = True

    def search(self, query, max_search=True):
        index = 0
        query_len = len(query)
        result = []
        while True:
            if index >= query_len:
                break
            match_results = []
            match_len = 0
            max_match_len = 0
            for start in range(index, query_len):
                check_word = query[index : start + 1]
                if check_word not in self.words_map:
                    break
                match_len += 1
                if self.words_map[check_word]:
                    match_results.append(check_word)
                    max_match_len = match_len

            if max_search is True and len(match_results) > 1:
                del match_results[:-1]
            for result_item in match_results:
                result.append({"pos": index, "word": result_item})

            if max_search and max_match_len > 0:
                index += max_match_len
            else:
                index += 1
        return result

    def remove_matched_words(self, query):
        result = self.search(query)
        remove_index_set = set()
        for item in result:
            for index in range(item["pos"], item["pos"] + len(item["word"])):
                remove_index_set.add(index)
        final_words = []
        for index, word in enumerate(list(query)):
            if index in remove_index_set:
                continue
            final_words.append(word)
        return "".join(final_words)


if __name__ == "__main__":
    dictionary_searcher = DictionarySearcher()
    for word in ["浙大", "浙江大学", "浙江"]:
        dictionary_searcher.add_word(word)

    sentence = "浙江省内最好的大学是浙江大学"

    result = dictionary_searcher.search(sentence, max_search=False)
    for item in result:
        print(item)
    dictionary_searcher = DictionarySearcher()
    dictionary_searcher.add_patterns("浙大\n浙江 浙江大学")
    print(dictionary_searcher.remove_matched_words(sentence))
    dictionary_searcher = DictionarySearcher()
    dictionary_searcher.add_patterns(" \n\n 浙江        浙江大学   \n .  浙大")
    print(dictionary_searcher.remove_matched_words(sentence))
