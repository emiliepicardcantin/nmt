from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


LANGUAGE = "english"
SENTENCES_COUNT = 3


if __name__ == "__main__":
    # url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # print(parser.document)

    # PlaintextParser(d)

    # # or for plain text files
    # # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    # stemmer = Stemmer(LANGUAGE)

    # summarizer = Summarizer(stemmer)
    # summarizer.stop_words = get_stop_words(LANGUAGE)

    # for sentence in summarizer(parser.document, SENTENCES_COUNT):
    #     print(sentence)

    print("Initialization")
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)

    print("Stop words")
    summarizer.stop_words = get_stop_words(LANGUAGE)

    with open('../datasets/news_summary/train.source', 'r') as f:
        source = next(f)

    with open('../datasets/news_summary/train.summary', 'r') as f:
        target_summary = next(f)

    print("Parsing document")
    parser = PlaintextParser.from_string(source, Tokenizer("english"))

    print("Getting summary...")
    summary = summarizer(parser.document, SENTENCES_COUNT)

    print()

    print("The original source : \n"+source+"\n")
    print("The expected summary : \n"+target_summary+"\n")

    print("Here is the summary : \n")
    for sentence in summary:
        print(sentence)