from math import log


class CountVectorizer:

    def fit_transform(self, corpus):
        self._bag_of_words = []
        self.corpus = corpus
        for elem in range(len(self.corpus)):
            self._bag_of_words.append(self.corpus[elem].split())
        self._bag_of_words = [y.lower() for x in self._bag_of_words for y in x]
        self._bag_of_words = sorted(list(set(self._bag_of_words)))
        matrix = []
        for phrase in self.corpus:
            phrase = phrase.split()
            matrix_line = []
            new_phrase = [word.lower() for word in phrase]
            for point in self._bag_of_words:
                matrix_line.append(new_phrase.count(point))
            matrix.append(matrix_line)
        self.matrix = matrix
        return matrix

    def get_feature_name(self):
        return self._bag_of_words


class TfIdfTransformer(CountVectorizer):

    def term_frequency(self, count_matrix):
        tf_matrix = []
        for elem in count_matrix:
            n = sum(elem)
            phrase_prob = []
            for num in elem:
                phrase_prob.append(round(num/n, 3))
            tf_matrix.append(phrase_prob)
        self.tf_matrix = tf_matrix
        return tf_matrix

    def idf_transform(self, count_matrix):
        num_docs = len(count_matrix)
        idf_matrix = []
        for i in range(len(count_matrix[0])):
            num_of_docs = 0
            for j in range(len(count_matrix)):
                num_of_inputs = count_matrix[j][i]
                if num_of_inputs > 0:
                    num_of_docs += 1
            idf_matrix.append(round((log((num_docs + 1) / (num_of_docs + 1)) + 1), 3))
        self.idf_matrix = idf_matrix
        return idf_matrix

    def fit_transform(self, count_matrix):
        tf = self.term_frequency(count_matrix)
        idf = self.idf_transform(count_matrix)
        answer = []
        for line in tf:
            answer.append([round(a*b, 3) for a, b in zip(line, idf)])
        return answer


class TfIdfVectorizer(CountVectorizer):

    def __init__(self):
        self.transformer = TfIdfTransformer()

    def fit_transform(self, corpus):
        count_matrix = super().fit_transform(corpus)
        return self.transformer.fit_transform(count_matrix)


corpus = [
    'Crock Pot Pasta Never boil pasta again',
    'Pasta Pomodoro Fresh ingredients Parmesan to taste'
]



# vectorizer = CountVectorizer()
# count_matrix = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_name())
# print(count_matrix)
#
#
# trans = TfIdfTransformer()
# print(trans.term_frequency(count_matrix))
# print(trans.idf_transform(count_matrix))
# print(trans.fit_transform(count_matrix))

vector = TfIdfVectorizer()
print(vector.fit_transform(corpus))
print(vector.get_feature_name())


