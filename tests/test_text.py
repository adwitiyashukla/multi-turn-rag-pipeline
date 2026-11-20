from turnrag.text import sentences, terms, truncate_words, words


def test_terms_drop_stopwords_and_stem():
    assert terms("The runners are running to the stadiums") == ["runner", "run", "stadium"]


def test_possessive_is_removed():
    assert words("John" + chr(0x2019) + "s car and Mary's bike") == [
        "john",
        "car",
        "and",
        "mary",
        "bike",
    ]


def test_sentences_merge_short_pieces():
    text = "Home\nThe Juno probe reached Jupiter in 2016. It measured the magnetic field."
    assert sentences(text) == [
        "Home The Juno probe reached Jupiter in 2016.",
        "It measured the magnetic field.",
    ]


def test_truncate_words():
    assert truncate_words("a b c d", 2) == "a b ..."
    assert truncate_words("a b", 5) == "a b"
