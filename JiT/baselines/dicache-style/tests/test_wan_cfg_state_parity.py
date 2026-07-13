def test_explicit_stream_id_matches_alternating_slot_mapping():
    sequence = ["cond", "uncond"] * 6
    explicit = {name: [] for name in {"cond", "uncond"}}
    slots = [[], []]
    for count, stream in enumerate(sequence):
        explicit[stream].append(count)
        slots[count % 2].append(count)
    assert explicit["cond"] == slots[0]
    assert explicit["uncond"] == slots[1]
