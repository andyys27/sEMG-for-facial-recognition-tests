import warnings
from src.protocol import parse_events, protocol_params


def test_parse_events_phases(events):
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no debe haber eventos desconocidos
        m = parse_events(events)
    assert m.loc[m.event == "hold_start", "phase"].eq("Gesto").all()
    assert m.loc[m.event.str.startswith("countdown"), "phase"].eq("Countdown").all()
    assert m.loc[m.event.isin(["session_start", "sequence_end"]), "phase"].isna().all()


def test_protocol_params(events):
    p = protocol_params(events)
    assert p["reps"] == 2 and p["gestures"] == 4 and p["hold_s"] == 3