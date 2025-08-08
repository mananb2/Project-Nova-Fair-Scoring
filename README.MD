# Project Nova: An Equitable Credit Scoring Engine for Grab's Partners

**A GrabHack Submission**

## 🎯 The Problem: The "Credit Invisible" Partner

[cite_start]Many of Grab's most reliable driver and merchant-partners have consistent income but lack the formal credit histories needed to access traditional financial products[cite: 5, 6]. [cite_start]This barrier prevents them from investing in their businesses—like upgrading a vehicle or expanding a food stall—severely limiting their economic mobility[cite: 7].

---

## ✨ Our Solution: The Nova Score

[cite_start]Project Nova introduces an equitable, data-driven credit score that evaluates a partner's creditworthiness based on their **proven performance and reliability within the Grab ecosystem**[cite: 9].

Our solution is a two-step process:
1.  [cite_start]**Predictive Score:** We first train an XGBoost model on a diverse set of simulated partner data (earnings, ratings, behavior, etc.) to generate a primary `nova_score`[cite: 13].
2.  **Fair Loan Approval:** We then use this score to power a **fairness-aware classification model** that decides on loan approvals. [cite_start]This model was trained using `fairlearn` to actively mitigate biases and ensure equitable outcomes across different partner groups[cite: 14].



---

## 📈 The Impact: Quantifying Financial Inclusion

[cite_start]Our fairness-mitigated model directly addresses the project's vision of creating a "fair, data-driven pathway to financial products"[cite: 10]. The results on the test data show a clear, positive impact on financial inclusion, especially for partners in previously underserved city tiers.

| City Tier | Baseline Approvals | **Mitigated Approvals** | Change |
| :-------: | :----------------: | :--------------------: | :----: |
|   Tier 1    |        150         |          **165** | **+15** |
|   Tier 2    |        110         |          **128** | **+18** |
|   Tier 3    |         85         |          **115** | **+30** |

*This translates to **63 more partners** gaining access to capital in our test set alone.*

---

## Transparency and Trust with XAI

A credit score shouldn't be a black box. [cite_start]To ensure **fairness, accountability, and transparency (FAT) in AI**[cite: 26], we used SHAP (SHapley Additive exPlanations) to make our model's decisions understandable.

**Why did a partner get a high score?**
*High weekly earnings, excellent ratings, and a high share of cashless transactions were key positive factors.*


**Why did a partner get a lower score?**
*High earnings volatility and low trip frequency were the primary negative drivers.*


This transparency helps build trust and gives partners actionable feedback on how to improve their financial standing.

---

## 🚀 Future Roadmap

* [cite_start]**Personalized Products:** Leverage the Nova Score to offer tailored insurance or other financial services[cite: 32, 33].
* **Real-time Monitoring:** Deploy the model with a monitoring system to ensure fairness and accuracy over time as data evolves.
* **Partner Dashboard:** Create an in-app dashboard for partners to view their Nova Score and understand the factors influencing it.

---

## Technical Details

**To run this project:**
1.  Clone the repository.
2.  Install the required packages: `pip install -r requirements.txt`
3.  Run the main script: `python project_nova_final.py`

The script will generate a `reports` folder containing all charts, SHAP plots, and a summary of the fairness metrics.