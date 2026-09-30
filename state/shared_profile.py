class SharedBootstrap:
    """共享 Bootstrap：只执行一次，三个房室复用"""

    @staticmethod
    def run(state):
        # 执行 data_introspection 和 plausibility_bounds
        # 结果写入 state.shared_profile
        # 三个房室分支从 shared_profile 读取，不重复执行
        pass


class CompartmentDispatcher:
    """房室分发器：管理 1/2/3 房室的并行执行"""

    COMPARTMENTS = [1, 2, 3]

    @staticmethod
    def dispatch(state, orchestrator):
        results = {}
        for n_cmt in CompartmentDispatcher.COMPARTMENTS:
            # 克隆 state，设置 locked_compartments
            branch_state = state.clone()
            branch_state.locked_compartments = n_cmt
            branch_state.shared_profile = state.shared_profile

            # 异步启动 NONMEM 执行
            future = orchestrator.run_async(branch_state)
            results[n_cmt] = future

        # 收集结果
        return {n: f.result() for n, f in results.items()}