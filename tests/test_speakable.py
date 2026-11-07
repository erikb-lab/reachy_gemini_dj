from reachy_brain.speakable import int_words, speakable


def test_team_numbers_and_years():
    assert speakable("Team 254") == "Team two fifty four"
    assert speakable("team 1678") == "team sixteen seventy eight"
    assert speakable("team 1005") == "team ten oh five"
    assert speakable("team 100") == "team one hundred"
    assert speakable("team 3600") == "team thirty six hundred"
    assert speakable("in 2026") == "in twenty twenty six"
    assert speakable("team 2000") == "team two thousand"


def test_plain_numbers_percent_ordinals_times():
    assert int_words(15) == "fifteen" and int_words(1234567) == "one million two hundred thirty four thousand five hundred sixty seven"
    assert speakable("Volume 15%") == "Volume fifteen percent"
    assert speakable("rank 3rd") == "rank third" and speakable("21st") == "twenty first"
    assert speakable("at 3:30 pm") == "at three thirty p m"
    assert speakable("at 7:05") == "at seven oh five"
    assert speakable("about 3.14") == "about three point one four"
    assert speakable("12,500 people") == "twelve thousand five hundred people"
    assert speakable("no digits here") == "no digits here"
