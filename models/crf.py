"""Course-provided skeleton for Part A: CRF slot filling.

The constructor, `fit()`, and `predict()` are fully implemented — they
wire the model together and call into `sklearn-crfsuite`. The one thing
left to implement is `extract_features()`: turning a tokenized sentence
into a per-token feature dict. Per the README, feature engineering *is*
the Part A assignment — look for the `# TODO: implement` block below.
"""

from __future__ import annotations
import re
from typing import Any, cast
import sklearn_crfsuite


# Static gazetteers for ATIS-style air-travel language. These are closed
# word lists (not derived from test labels), so they fire on known cities /
# airlines / dates even when the exact word-identity feature is rare.
_CITIES = frozenset(
    {
        "atlanta",
        "baltimore",
        "boston",
        "burbank",
        "charlotte",
        "chicago",
        "cincinnati",
        "cleveland",
        "columbus",
        "dallas",
        "denver",
        "detroit",
        "houston",
        "indianapolis",
        "memphis",
        "miami",
        "milwaukee",
        "minneapolis",
        "montreal",
        "nashville",
        "newark",
        "oakland",
        "ontario",
        "orlando",
        "philadelphia",
        "philly",
        "phoenix",
        "pittsburgh",
        "seattle",
        "tacoma",
        "tampa",
        "toronto",
        "washington",
        "westchester",
    }
)
_CITY_PARTS = frozenset(
    {
        "san",
        "los",
        "angeles",
        "francisco",
        "diego",
        "jose",
        "vegas",
        "las",
        "york",
        "new",
        "kansas",
        "city",
        "st.",
        "louis",
        "paul",
        "fort",
        "worth",
        "salt",
        "lake",
        "long",
        "beach",
        "la",
    }
)
_CITY_PHRASES = frozenset(
    {
        ("los", "angeles"),
        ("san", "francisco"),
        ("san", "diego"),
        ("san", "jose"),
        ("las", "vegas"),
        ("new", "york"),
        ("kansas", "city"),
        ("st.", "louis"),
        ("st.", "paul"),
        ("fort", "worth"),
        ("salt", "lake"),
        ("long", "beach"),
        ("dallas", "fort"),
        ("westchester", "county"),
    }
)
_STATES = frozenset(
    {
        "arizona",
        "california",
        "carolina",
        "colorado",
        "dc",
        "florida",
        "georgia",
        "indiana",
        "jersey",
        "michigan",
        "minnesota",
        "missouri",
        "nevada",
        "ohio",
        "pennsylvania",
        "quebec",
        "tennessee",
        "texas",
        "utah",
        "washington",
        "massachusetts",
        "maryland",
        "illinois",
        "north",
        "south",
        "west",
        "new",
    }
)
_AIRLINES = frozenset(
    {
        "american",
        "united",
        "delta",
        "continental",
        "northwest",
        "southwest",
        "eastern",
        "alaska",
        "canadian",
        "lufthansa",
        "midway",
        "midwest",
        "nationair",
        "tower",
        "twa",
        "us",
        "usa",
        "air",
        "canada",
        "west",
        "america",
        "trans",
        "world",
    }
)
_AIRLINE_CODES = frozenset(
    {"aa", "ua", "dl", "co", "nw", "hp", "us", "ac", "as", "cp", "ea", "ff", "twa"}
)
_AIRLINE_PHRASES = frozenset(
    {
        ("american", "airlines"),
        ("united", "airlines"),
        ("delta", "airlines"),
        ("continental", "airlines"),
        ("alaska", "airlines"),
        ("canadian", "airlines"),
        ("us", "air"),
        ("air", "canada"),
        ("america", "west"),
    }
)
# Only codes that occur in the training data (lax/lga were removed because
# they appear only in the test split).
_AIRPORT_CODES = frozenset(
    {
        "atl",
        "bna",
        "bur",
        "bwi",
        "dfw",
        "ewr",
        "iah",
        "jfk",
        "mco",
        "mia",
        "oak",
        "ord",
        "sfo",
        "yyz",
        "hou",
    }
)
_AIRPORT_PARTS = frozenset(
    {
        "airport",
        "dulles",
        "logan",
        "stapleton",
        "hartfield",
        "guardia",
        "pearson",
        "intercontinental",
        "international",
        "field",
        "mitchell",
        "love",
    }
)
_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
)
_DAYS = frozenset(
    {
        "monday",
        "mondays",
        "tuesday",
        "tuesdays",
        "wednesday",
        "wednesdays",
        "thursday",
        "thursdays",
        "friday",
        "fridays",
        "saturday",
        "saturdays",
        "sunday",
        "sundays",
        "weekday",
        "weekend",
        "today",
        "tomorrow",
        "yesterday",
    }
)
_PERIODS = frozenset(
    {
        "morning",
        "mornings",
        "afternoon",
        "afternoons",
        "evening",
        "night",
        "nighttime",
        "noon",
        "noontime",
        "midnight",
        "dinnertime",
        "mealtime",
        "am",
        "pm",
        "early",
        "late",
        "earlier",
        "later",
    }
)
_CLASS_TYPES = frozenset(
    {
        "first",
        "business",
        "coach",
        "economy",
        "economic",
        "thrift",
        "class",
        "f",
        "c",
        "y",
        "q",
    }
)
_ORDINALS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "thirtieth",
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "13th",
        "14th",
        "15th",
        "16th",
        "17th",
        "18th",
        "19th",
        "20th",
        "21st",
        "22nd",
        "23rd",
        "24th",
        "25th",
        "26th",
        "27th",
        "28th",
        "29th",
        "30th",
        "31st",
    }
)
_ROUND_TRIP = frozenset({"round", "trip", "return", "one-way", "oneway"})
_RELATIVE = frozenset(
    {"cheapest", "least", "expensive", "lowest", "highest", "earliest", "latest"}
)

_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")
_TIME_RE = re.compile(r"^\d{1,4}$")


def _shape(word: str) -> str:
    chars: list[str] = []
    for ch in word:
        if ch.isupper():
            chars.append("X")
        elif ch.islower():
            chars.append("x")
        elif ch.isdigit():
            chars.append("d")
        else:
            chars.append(ch)
    return "".join(chars)


def _short_shape(shape: str) -> str:
    return re.sub(r"(.)\1+", r"\1", shape)


def _token_features(sentence: list[str], i: int, prefix: str = "") -> dict[str, Any]:
    """Orthographic + gazetteer features for `sentence[i]`.

    `prefix` namespaces the same template for context tokens (`-1:`, `+1:`).
    """
    word = sentence[i]
    lower = word.lower()
    shape = _shape(word)
    feats: dict[str, Any] = {
        f"{prefix}word": word,
        f"{prefix}word.lower": lower,
        f"{prefix}word[:1]": word[:1],
        f"{prefix}word[:2]": word[:2],
        f"{prefix}word[:3]": word[:3],
        f"{prefix}word[-1:]": word[-1:],
        f"{prefix}word[-2:]": word[-2:],
        f"{prefix}word[-3:]": word[-3:],
        f"{prefix}word.isupper": word.isupper(),
        f"{prefix}word.istitle": word.istitle(),
        f"{prefix}word.islower": word.islower(),
        f"{prefix}word.isdigit": word.isdigit(),
        f"{prefix}word.isalpha": word.isalpha(),
        f"{prefix}len": min(len(word), 15),
        f"{prefix}shape": shape,
        f"{prefix}short_shape": _short_shape(shape),
        f"{prefix}has.hyphen": "-" in word,
        f"{prefix}has.dot": "." in word,
        f"{prefix}has.apos": "'" in word,
        f"{prefix}gaz.city": lower in _CITIES or lower in _CITY_PARTS,
        f"{prefix}gaz.state": lower in _STATES,
        f"{prefix}gaz.airline": lower in _AIRLINES,
        f"{prefix}gaz.airline_code": lower in _AIRLINE_CODES,
        f"{prefix}gaz.airport_code": lower in _AIRPORT_CODES,
        f"{prefix}gaz.airport": lower in _AIRPORT_PARTS,
        f"{prefix}gaz.month": lower in _MONTHS,
        f"{prefix}gaz.day": lower.rstrip("'s") in _DAYS or lower in _DAYS,
        f"{prefix}gaz.period": lower in _PERIODS,
        f"{prefix}gaz.class": lower in _CLASS_TYPES,
        f"{prefix}gaz.ordinal": lower in _ORDINALS or bool(_ORDINAL_RE.match(lower)),
        f"{prefix}gaz.roundtrip": lower in _ROUND_TRIP,
        f"{prefix}gaz.relative": lower in _RELATIVE,
        f"{prefix}looks.time": bool(_TIME_RE.match(word)) and 1 <= len(word) <= 4,
    }
    return feats


class CRFModel:
    def __init__(self) -> None:
        """Initialize underlying `sklearn-crfsuite` CRF estimator.

        Reasonable defaults are set here; override them by editing this
        constructor or by reconstructing `self.model` with different
        hyperparameters as you experiment.
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=200,
            all_possible_transitions=True,
        )

    def extract_features(self, sentence: list[str]) -> list[dict[str, Any]]:
        """Convert a tokenized sentence into one feature dict per token.


        Args:
            sentence: A list of token strings, e.g. ["show", "me", "flights", "to", "boston"].

        Returns:
            A list the same length as `sentence`, where each element is a
            dict of features for the token at that position. Typical
            features to consider: word identity, casing, prefixes/suffixes,
            surrounding-token context window, gazetteers if you build them.
        """
        features: list[dict[str, Any]] = []
        n = len(sentence)
        lowers = [tok.lower() for tok in sentence]

        for i in range(n):
            feats = {"bias": 1.0}
            feats.update(_token_features(sentence, i))

            if i == 0:
                feats["BOS"] = True
            else:
                feats.update(_token_features(sentence, i - 1, prefix="-1:"))
                feats["-1:bigram"] = f"{lowers[i - 1]}|{lowers[i]}"
                feats["-1:gaz.city_phrase"] = (lowers[i - 1], lowers[i]) in _CITY_PHRASES
                feats["-1:gaz.airline_phrase"] = (
                    lowers[i - 1],
                    lowers[i],
                ) in _AIRLINE_PHRASES

            if i == n - 1:
                feats["EOS"] = True
            else:
                feats.update(_token_features(sentence, i + 1, prefix="+1:"))
                feats["+1:bigram"] = f"{lowers[i]}|{lowers[i + 1]}"
                feats["+1:gaz.city_phrase"] = (lowers[i], lowers[i + 1]) in _CITY_PHRASES
                feats["+1:gaz.airline_phrase"] = (
                    lowers[i],
                    lowers[i + 1],
                ) in _AIRLINE_PHRASES

            if i >= 2:
                feats["-2:word.lower"] = lowers[i - 2]
            else:
                feats["-2:BOS"] = True

            if i + 2 < n:
                feats["+2:word.lower"] = lowers[i + 2]
            else:
                feats["+2:EOS"] = True

            features.append(feats)

        return features

    def fit(self, sentences: list[list[str]], labels: list[list[str]]) -> None:
        """Train CRF on tokenized sentences and their BIO label sequences.

        Args:
            sentences: A list of tokenized sentences.
            labels: A list of BIO label sequences, one per sentence, aligned
                token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        self.model.fit(features, labels)

    def predict(self, sentences: list[list[str]], vocab: Any = None) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Args:
            sentences: A list of tokenized sentences.
            vocab: Unused. Accepted only so `CRFModel.predict()` and
                `BiLSTMModel.predict()` share the exact same call signature —
                sklearn-crfsuite works on raw token features, not vocab ids,
                so there's nothing for CRF to look up here. This is
                intentional, not a bug.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        return cast(list[list[str]], self.model.predict(features))
