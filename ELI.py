@@ -0,0 +1,29 @@
@@ -0,0 +1,28 @@
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize
np.random.seed(42)
df = pd.DataFrame({
    'cibil_score': np.random.randint(300,900,300),
    'loan_amount': np.random.randint(50000,2000000,300)
})
df['eligible'] = ((df['cibil_score']>650) & (df['loan_amount']<1500000)).astype(int)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
X = np.column_stack([np.ones(len(df)), df['cibil_score']/1000, df['loan_amount']/1e6])
y = df['eligible'].values
def cost(t):
    h=np.clip(sigmoid(X@t),1e-15,1-1e-15)
    return -np.mean(y*np.log(h)+(1-y)*np.log(1-h))
theta = minimize(cost, np.zeros(3), method='BFGS').x

plt.figure(figsize=(8,6))
plt.scatter(df[df['eligible']==0]['cibil_score'], df[df['eligible']==0]['loan_amount'], c='red', label='Not Eligible', alpha=0.6)
plt.scatter(df[df['eligible']==1]['cibil_score'], df[df['eligible']==1]['loan_amount'], c='green', label='Eligible', alpha=0.6)
x_vals=np.linspace(300,900,100)
y_vals = -(theta[0] + theta[1]*x_vals/1000) *1e6 / theta[2]
plt.plot(x_vals, y_vals, 'b--', label='Decision Boundary')
plt.xlabel('CIBIL Score (X-axis)'); plt.ylabel('Loan Amount (Y-axis)')
plt.title('Loan Eligibility Prediction using Logistic Regression')
plt.legend(); plt.ylim(0,2100000)
plt.savefig('loan_eligibility_plot.png')
plt.show()
