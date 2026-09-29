import json

from resremd import testsystems
from resremd.cli import build_parser, flag, main, template
from resremd.options import GENERATE, IMPORT, RUN, SCHEMAS


def test_run_and_summary_from_the_command_line(tmp_path, capsys):
    testsystems.write_double_well_reservoir(tmp_path / "res", kind="boltzmann",
                                       n_frames=200, temperature_K=520.0)
    prepared = testsystems.double_well()
    import openmm
    from openmm import app

    d = tmp_path / "setup"
    d.mkdir()
    (d / "system.xml").write_text(openmm.XmlSerializer.serialize(
        prepared.system))
    ctx = openmm.Context(prepared.system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(prepared.positions)
    (d / "state.xml").write_text(openmm.XmlSerializer.serialize(
        ctx.getState(getPositions=True)))
    with open(d / "topology.pdb", "w") as fh:
        app.PDBFile.writeFile(prepared.topology, prepared.positions * 10, fh)
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"temperatures_K": [300, 400],
                                  "platform": "Reference"}))
    code = main(["-q", "run", "--config", str(config), "--prepared", str(d),
                 "--reservoir", str(tmp_path / "res"), "--output",
                 str(tmp_path / "run"), "--production-steps", "1000",
                 "--exchange-interval-steps", "50", "--no-minimize",
                 "--equilibration-ns", "0", "--save-states", "lowest",
                 "--random-seed", "1"])
    assert code == 0
    capsys.readouterr()
    assert main(["summary", str(tmp_path / "run"), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["cycles"] == 20
    assert summary["temperatures_K"] == [300.0, 400.0]


def test_errors_exit_with_a_code(tmp_path, capsys):
    assert main(["-q", "run", "--output", str(tmp_path / "x")]) == 2


def test_ladder_and_options(capsys):
    assert main(["ladder", "--temperature-min-K", "300",
                 "--temperature-max-K", "400", "--n-replicas", "3"]) == 0
    assert capsys.readouterr().out.split() == ["300.00", "346.41", "400.00"]
    assert main(["options", "generate"]) == 0
    assert "frame_interval_steps: 5000" in capsys.readouterr().out


def test_every_option_has_a_flag_and_the_flags_parse():
    parser = build_parser()
    for schema, argv in ((RUN, ["run"]), (GENERATE, ["reservoir", "generate"]),
                         (IMPORT, ["reservoir", "import"])):
        text = parser._subparsers._group_actions[0].choices
        sub = text[argv[0]] if len(argv) == 1 else \
            text[argv[0]]._subparsers._group_actions[0].choices[argv[1]]
        flags = {s for a in sub._actions for s in a.option_strings}
        for option in schema.options:
            assert flag(option) in flags, option.name


def test_template_names_every_option():
    for schema in SCHEMAS.values():
        text = template(schema)
        for option in schema.options:
            assert f"{option.name}:" in text


def test_list_settings_of_mappings_and_lists_are_read_as_json():
    args = build_parser().parse_args([
        "reservoir", "generate", "--prepared", "x", "--output", "r",
        "--temperature-K", "500", "--duration-ns", "1",
        "--bias-torsions", '{"atoms": [0, 1, 2, 3], "energy": "theta"}',
        "--convergence-torsions", "[0, 1, 2, 3]", "[1, 2, 3, 4]"])
    assert args.bias_torsions == [{"atoms": [0, 1, 2, 3], "energy": "theta"}]
    assert args.convergence_torsions == [[0, 1, 2, 3], [1, 2, 3, 4]]
