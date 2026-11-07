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


def test_team_style_only_after_the_word_team():
    assert speakable("We scored 128 points") == "We scored one hundred twenty eight points"
    assert speakable("pit 1234") == "pit one thousand two hundred thirty four"
    assert speakable("teams 254 and 1678") == "teams two fifty four and sixteen seventy eight"
    assert speakable("Team 254 beat 118") == "Team two fifty four beat one hundred eighteen"
