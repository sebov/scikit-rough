# Staging Area

This directory holds propositions that are **pending verification** and not part of the verified knowledge base.

## Contents

- `bireduct-attrs-subset-form-bireduct.md`: For bireduct $(X, B)$ and $B' \subseteq B$, exists $X'$ such that $(X', B')$ is bireduct. **Issue**: proof in source (`tmp/pub/main.tex:331-338`) is incomplete (Polish, cuts off mid-argument). General case unverified.
- `bireduct-equiv-classes-geq-bplus1.md`: $|X/B| \geq |B| + 1$ for any bireduct. **Issue**: induction proof depends on unverified `prop-bireduct-attrs-subset-form-bireduct`.
- `bireduct-desc-len-geq-bplus1-squared.md`: $BireductDescLen(X, B) \geq (|B| + 1)^2$. **Issue**: depends on unverified `prop-bireduct-equiv-classes-geq-bplus1`.

## Dependency Chain

```
prop-bireduct-attrs-subset-form-bireduct (unverified)
  └─> prop-bireduct-equiv-classes-geq-bplus1 (depends on above in induction)
        └─> prop-bireduct-desc-len-geq-bplus1-squared (depends on above)
```

## Resolution

Once `prop-bireduct-attrs-subset-form-bireduct` is either proven or disproven:
- If proven: move all 3 back to `propositions/`, update `index.md`
- If disproven: revise or remove dependent propositions, update proofs accordingly
