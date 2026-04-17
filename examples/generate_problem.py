import yaml
import random

def generate_knapsack_yaml(n=50, seed=42):
    random.seed(seed)
    
    # 随机生成物品价值和重量
    values = [random.randint(10, 100) for _ in range(n)]
    weights = [random.randint(5, 50) for _ in range(n)]
    # 设置容量为总重量的 40%
    capacity = int(sum(weights) * 0.4)

    # 构建 problem 部分
    problem = {
        "objective": {
            "sense": "max",
            "coefficients": [float(v) for v in values]
        },
        "constraints": []
    }

    # 1. 添加容量约束: sum(w_i * x_i) <= W
    problem["constraints"].append({
        "name": "capacity_limit",
        "coefficients": [float(w) for w in weights],
        "sense": "<=",
        "rhs": float(capacity)
    })

    # 2. 添加 0-1 约束: x_i <= 1 (针对每个变量)
    for i in range(n):
        coeffs = [0.0] * n
        coeffs[i] = 1.0
        problem["constraints"].append({
            "name": f"ub_x{i}",
            "coefficients": coeffs,
            "sense": "<=",
            "rhs": 1.0
        })

    # 构建 config 部分
    config = {
        "is_integer": True,
        "integer_indices": list(range(n)),
        "lp_method": "primal",
        "epsilon": 1.0e-9,
        "max_iterations": 10000,
        "max_nodes": 50000,
        "use_rounding_heuristic": True,
        "use_gomory_cuts": True,
        "max_gomory_cuts_per_node": 5
    }

    full_config = {
        "problem": problem,
        "config": config
    }

    with open("problem_knapsack_50.yaml", "w") as f:
        yaml.dump(full_config, f, sort_keys=False)
    
    print(f"成功生成 50 变量背包问题配置文件。容量: {capacity}")

if __name__ == "__main__":
    generate_knapsack_yaml()