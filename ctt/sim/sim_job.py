import covid19sim.run


def main():
    """Hot wraps `covid19sim.run.main`."""
    conf = covid19sim.run.main()
    print(f"@ctt.sim.sim_server:{conf['outdir']}")


if __name__ == '__main__':
    main()