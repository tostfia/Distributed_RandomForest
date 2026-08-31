import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = np.random.rand(50, 3)
y = np.random.randint(0, 2, 50)
rf = RandomForestClassifier(n_estimators=3, oob_score=True, random_state=0, bootstrap=True)
rf.fit(X, y)

odf = rf.oob_decision_function_
valid_mask = odf.sum(axis=1) != 0

# Accuracy "corretta" (solo righe valide)
pred_corretta = np.argmax(odf[valid_mask], axis=1)
acc_corretta = accuracy_score(y[valid_mask], pred_corretta)

# Accuracy "naive" su TUTTE le righe (comportamento di argmax senza controllo)
pred_naive = np.argmax(odf, axis=1)
acc_naive = accuracy_score(y, pred_naive)

print(f"rf.oob_score_ (nativo sklearn) : {rf.oob_score_:.5f}")
print(f"Accuracy corretta (valid_mask) : {acc_corretta:.5f}")
print(f"Accuracy naive (tutte le righe): {acc_naive:.5f}")