import emergence as em

if __name__ == '__main__':
    model_cfg = {
    }
    model = em.model.Binary(model_cfg)
    ap = em.AgentPair(model)
    ap.train(verbose=True)
    ap.test(verbose=True)
