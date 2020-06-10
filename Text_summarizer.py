from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# Open and read the input file
f = open("Input.txt",encoding="utf8")
text_str = f.read()
print('Original Text:\n',text_str)

def _create_frequency_table(text_string) -> dict:
# Create a dictionary for the word frequency table by removing the stopwords
    
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    # Stemmer - an algorithm to bring words to its root word.
    ps = PorterStemmer()
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
# score a sentence by its words
# Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
# Find the average score from the sentence value dictionary

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    
    return summary


def run_summarization(text):
    
    freq_table = _create_frequency_table(text)
    
    sentences = sent_tokenize(text)
    
    sentence_scores = _score_sentences(sentences, freq_table)
    
    threshold = _find_average_score(sentence_scores)
    
    summary = _generate_summary(sentences, sentence_scores, 1 * threshold)
    return summary

if __name__ == '__main__':
    result = run_summarization(text_str)
    print('\nSummary:\n',result)
    
f = open("Summary.txt","w+")
f.write(result)
f.close() 
