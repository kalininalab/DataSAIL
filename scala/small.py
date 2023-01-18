import numpy as np

cyclophilins = ["4FRV_A", "4FRU_A", "1CYN_A", "8CU5_A", "3BT8_A", "6L2B_A", "4S1J_A", "2HAQ_A", "4S1E_A", "1XO7_A"]
kinases = ["6B3J_R", "7RTB_R", "7FIM_R", "7X8R_R", "7DUQ_R", "6VCB_R", "5NX2_A", "7S15_R", "6LN2_A", "5VAI_R"]
gpcrs = ["5C01_A", "4OLI_A", "3ZON_A", "4WOV_A", "5TKD_A", "7AX4_A", "6NZR_A", "4L00_A", "4L01_A"]

with open("../tests/data/pipeline/drugs.tsv", "r") as drug_input:
    drugs = [line.split("\t")[0] for line in drug_input.readlines()]

proteins = cyclophilins + kinases + gpcrs
drugs_s1, drugs_s2 = drugs[:33], drugs[:33]
prots_s1, prots_s2 = proteins[:20], proteins[20:]
prot_sim = np.zeros((len(proteins), len(proteins)))
drug_sim = np.zeros((len(drugs), len(drugs)))

for i in range(len(drugs)):
    drug_sim[i, i] = 1
    for j in range(i + 1, len(drugs)):
        drug_sim[i, j] = max(min(np.random.normal(0.85 if (i < 33) == (j < 33) else 0.15, 0.1), 1), 0)
        drug_sim[j, i] = drug_sim[i, j]

for i in range(len(proteins)):
    prot_sim[i, i] = 1
    for j in range(i + 1, len(proteins)):
        prot_sim[i, j] = max(min(np.random.normal(0.85 if (i < 20) == (j < 20) else 0.15, 0.1), 1), 0)
        prot_sim[j, i] = prot_sim[i, j]

with open("../tests/data/pipeline/inter.tsv", "w") as inter:
    for d, drug in enumerate(drugs):
        for p, protein in enumerate(proteins):
            if ((d < 33) == (p < 20) and np.random.random() > 0.1) or \
                    ((d < 33) == (p < 20) and np.random.random() < 0.1):
                print(drug, protein, 1, sep="\t", file=inter)


with open("../tests/data/pipeline/prot_sim.tsv", "w") as prot_sim_file:
    for i, prot in enumerate(proteins):
        print(f"{prot}\t", end="", file=prot_sim_file)
        for x in prot_sim[i, :]:
            print(f"{x:3f}\t", end="", file=prot_sim_file)
        print(file=prot_sim_file)

with open("../tests/data/pipeline/drug_sim.tsv", "w") as drug_sim_file:
    for i, drug in enumerate(drugs):
        print(f"{drug}", end="", file=drug_sim_file)
        for x in drug_sim[i, :]:
            print(f"\t{x:3f}", end="", file=drug_sim_file)
        print(file=drug_sim_file)
