from reachy_brain.speakable import int_words, speakable


def test_team_numbers_are_read_digit_by_digit_and_years_as_pairs():
    assert speakable("Team 254") == "Team two five four"
    assert speakable("team 1678") == "team one six seven eight"
    assert speakable("team 1005") == "team one zero zero five"
    assert speakable("team 100") == "team one zero zero"
    assert speakable("team 3600") == "team three six zero zero"
    assert speakable("in 2026") == "in twenty twenty six"
    assert speakable("team 2000") == "team two zero zero zero"


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
    assert speakable("teams 254 and 1678") == "teams two five four and one six seven eight"
    assert speakable("Team 254 beat 118") == "Team two five four beat one hundred eighteen"
