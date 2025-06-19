from das_anomaly.psd import PSDGenerator


def test_run_calls_plot_spec(cfg, patched_plot_psd, dummy_patch):
    gen = PSDGenerator(cfg)
    gen.run()

    # Should have been called 14 times
    # (len(sub_sp.chunk(time=time_window, overlap=time_overlap)) == 14)
    assert patched_plot_psd.call_count == 14

    # Inspect for expected arguments
    args, kwargs = patched_plot_psd.call_args_list[0]

    patch_arg, min_f, max_f, sr, title = args
    assert isinstance(patch_arg, type(dummy_patch))
    assert sr == 23787
    assert min_f == 0

    assert kwargs["output_rank"] == 0
    assert kwargs["fig_path"] == cfg.psd_path
    assert kwargs["dpi"] == cfg.dpi
