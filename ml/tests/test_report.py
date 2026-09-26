from rifteye_ml import report, spike


def test_report_from_a_demo_run(tmp_path):
    out = tmp_path / "m0-demo.csv"
    args = ["demo", "--cards", "12", "--heights", "40,120", "--bitrates", "3000", "--encoder", "colorgrid",
            "--encoder", "dhash", "--out", str(out)]
    assert spike.main(args) == 0
    rows = report.load(out)
    table = report.markdown_table(rows, "3000")
    assert table.splitlines()[0] == "| Encoder | Rotation | 40 px | 120 px |"
    assert len(table.splitlines()) == 2 + 4  # 2 encoders × 2 rotation modes
    svg = report.svg_chart(rows, "3000", "search", title="demo")
    assert svg.count("<polyline") == 2 and "colorgrid16" in svg and "1280x720" in svg
    assert report.main(["--csv", str(out), "--svg", str(tmp_path / "c.svg"), "--bitrate", "3000"]) == 0
    assert (tmp_path / "c.svg").read_text().startswith("<svg")
