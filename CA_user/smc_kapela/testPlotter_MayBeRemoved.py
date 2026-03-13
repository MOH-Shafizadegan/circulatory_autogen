import matplotlib.pyplot as plt  
import os  
  
test_dir = r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen\param_id_output\genetic_algorithm_smc_kapela_VoltageClamp_obs_20260305\plots_param_id"  
  
plt.plot([1, 2, 3])  
try:  
    plt.savefig(os.path.join(test_dir, "test.eps"))  
    print("EPS works")  
except:  
    print("EPS failed, trying PDF")  
    plt.savefig(os.path.join(test_dir, "test.pdf"))  
plt.close()