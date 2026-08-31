from huggingface_hub import hf_hub_download
import torch
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase import Atoms
from ase.io import read, write
from ase.filters import ExpCellFilter
from ase.optimize import LBFGS,FIRE


import os
import glob
import yaml

class MemoryManager:
    def __init__(self, output_dir):
        self.yaml_path = os.path.join(output_dir, "status.yaml")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.data = self.load()

    def load(self):
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def save(self):
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.data, f, allow_unicode=True, default_flow_style=False)

    def get_status(self, name_without_ext):
        return self.data.get(name_without_ext, {})

    def update_task(self, name_without_ext, status, geo_params=None):
        if name_without_ext not in self.data:
            self.data[name_without_ext] = {}
        self.data[name_without_ext]['status'] = status
        
        if geo_params is not None:
            if isinstance(geo_params, list):
                params_to_save = []
                for p in geo_params:
                    p_copy = p.copy()
                    if 'ase_model' in p_copy and hasattr(p_copy['ase_model'], '__name__'):
                        p_copy['ase_model'] = p_copy['ase_model'].__name__
                    params_to_save.append(p_copy)
                self.data[name_without_ext]['geo_params'] = params_to_save
            else:
                params_to_save = geo_params.copy()
                if 'ase_model' in params_to_save and hasattr(params_to_save['ase_model'], '__name__'):
                    params_to_save['ase_model'] = params_to_save['ase_model'].__name__
                self.data[name_without_ext]['geo_params'] = params_to_save
        self.save()

class Model:
    def __init__(self, filename="pretrained/uma-s-1p1.pt"):
        self.path = hf_hub_download(
            repo_id="zhilong777/CP-catalysis-env", 
            filename=filename
        )

    def get_model(self):
        print("使用模型: " + self.path)
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available() :
         print(f"Current device: {torch.cuda.get_device_name(0)}")
         print(f"CUDA Version: {torch.version.cuda}")
         print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            print(f"CUDA is not available") 

        return load_predict_unit(
            path=self.path,
            device="cuda"
        )
        



class FileHandler:
    def __init__(self, input_dir="DATA/", output_dir="OUTPUT/"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_file_paths(self, geo_params=None, memory_manager=None):
        """
        扫描输入文件夹，并为每个文件准备对应的输出文件夹及文件路径。
        返回包含 (输入cif路径, 输出cif路径, 输出traj路径, 无后缀文件名) 的列表。
        """
        search_pattern = os.path.join(self.input_dir, "*.cif")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"在 {self.input_dir} 中没有找到任何 .cif 文件")
            return []

        path_list = []
        for file_path in files:
            filename = os.path.basename(file_path)
            name_without_ext, _ = os.path.splitext(filename)
            
            # 判断是否需要跳过计算
            if geo_params and memory_manager:
                task_info = memory_manager.get_status(name_without_ext)
                current_status = task_info.get('status', 'pending')
                saved_params = task_info.get('geo_params', {})
                
                if isinstance(geo_params, list):
                    current_params = []
                    for p in geo_params:
                        p_copy = p.copy()
                        if 'ase_model' in p_copy and hasattr(p_copy['ase_model'], '__name__'):
                            p_copy['ase_model'] = p_copy['ase_model'].__name__
                        current_params.append(p_copy)
                else:
                    current_params = geo_params.copy()
                    if 'ase_model' in current_params and hasattr(current_params['ase_model'], '__name__'):
                        current_params['ase_model'] = current_params['ase_model'].__name__

                params_changed = (saved_params and saved_params != current_params)
                
                if current_status == 'completed' and not params_changed:
                    print(f"[{filename}] 已计算且参数未变，跳过...")
                    continue
                    
                if current_status == 'recalculate' or params_changed:
                    print(f"[{filename}] 参数变更或标记为重算，准备重新计算...")
                else:
                    print(f"[{filename}] 状态为 {current_status}，准备计算...")
            
            # 为每个文件建立专属的输出文件夹，按名字存放
            file_output_dir = os.path.join(self.output_dir, name_without_ext)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)
                
            output_cif = os.path.join(file_output_dir, f"optimized_{name_without_ext}.cif")
            output_traj = os.path.join(file_output_dir, "optimization.traj")
            
            path_list.append((file_path, output_cif, output_traj, name_without_ext))
            
        return path_list


class Geo:
    def __init__(self, input_atoms, model, model_type="omc", ase_model=FIRE, steplength=0.05, f_max=5, use_cell_filter=False, mask=None):
        # 记录文件路径，方便后续生成默认输出路径
        self.input_path = None
        if isinstance(input_atoms, str):
            self.input_path = input_atoms
            self.input_atoms = read(input_atoms) 
        else:
            self.input_atoms = input_atoms
            
        self.model = model
        self.model_type = model_type
        self.ase_model = ase_model 
        self.steplength = steplength
        self.f_max = f_max
        self.use_cell_filter = use_cell_filter
        self.mask = mask if mask is not None else [1, 1, 1, 0, 0, 0]

    def run(self, output_path=None, traj_path=None):
       
        if output_path is None:
            if self.input_path:
                input_filename = os.path.basename(self.input_path)
                input_name_without_ext, input_ext = os.path.splitext(input_filename)
                output_path = os.path.join(os.path.dirname(self.input_path), f"optimized_{input_name_without_ext}{input_ext}")    
            else:
                output_path = "optimized_output.cif"

        if traj_path is None:
            traj_path = f"optimization_{self.model_type}.traj"

        calc = FAIRChemCalculator(self.model, task_name=self.model_type)
        self.input_atoms.calc = calc
        
        print(f"该材料包含 {len(self.input_atoms)} 个原子。")
        print(f"初始晶胞大小: {self.input_atoms.cell.cellpar()}")
        print(f"初始总能量: {self.input_atoms.get_potential_energy():.4f} eV")
        
        # 判断是否使用 ExpCellFilter 进行晶胞优化
        if self.use_cell_filter:
            target_atoms = ExpCellFilter(self.input_atoms, mask=self.mask)
            print(f"使用 ExpCellFilter 进行晶胞优化 (mask={self.mask})")
        else:
            target_atoms = self.input_atoms
            print("进行普通原子位置优化 (不优化晶胞)")

        opt = self.ase_model(target_atoms, trajectory=traj_path, maxstep=self.steplength)
        opt.run(fmax=self.f_max)
        
        write(output_path, self.input_atoms)
        print(f"优化后的结构已成功输出并保存为: {output_path}")

        return self.input_atoms

if __name__ == "__main__":
    # ------------------ 全局配置 ------------------
    INPUT_DIR = "DATA/"
    OUTPUT_DIR = "OUTPUT/"
    # ---------------------------------------------
    
    # 1. 初始化并获取模型
    UMAmodel = Model()
    model = UMAmodel.get_model()
    
    # 2. 初始化记忆管理和路径管理
    memory_manager = MemoryManager(output_dir=OUTPUT_DIR)
    file_handler = FileHandler(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    
    # 统一管理多步连续优化的 geo 参数
    geo_steps = [
        {
            "model_type": "omc",
            "ase_model": FIRE,
            "steplength": 0.05,
            "f_max": 5,
            "use_cell_filter": False,
            "mask": [1, 1, 1, 1, 1, 1]
        },
        {
            "model_type": "omc",
            "ase_model": FIRE,
            "steplength": 0.05,
            "f_max": 3,
            "use_cell_filter": True,
            "mask": [1, 1, 1, 0, 0, 0]
        },
        {
            "model_type": "omc",
            "ase_model": LBFGS,
            "steplength": 0.02,
            "f_max": 0.05,
            "use_cell_filter": True,
            "mask": [1, 1, 1, 0, 0, 0]
        }
    ]
    
    tasks = file_handler.get_file_paths(geo_params=geo_steps, memory_manager=memory_manager)
    
    # 3. 遍历每个文件，进行多步连续计算和状态管理
    for input_cif, output_cif, output_traj, name_without_ext in tasks:
        print(f"\n========== 开始处理: {input_cif} ==========")
        
        try:
            current_atoms = input_cif # 第一步传入文件路径
            for step_idx, params in enumerate(geo_steps):
                print(f"\n--- 执行第 {step_idx + 1} 步优化 ---")
                
                # 为了防止多步优化互相覆盖轨迹文件，在名称里加入步骤号
                step_traj = output_traj.replace(".traj", f"_step{step_idx + 1}.traj")
                
                geo_runner = Geo(
                    input_atoms=current_atoms, 
                    model=model, 
                    **params
                )
                
                # 每次执行都会把结果保存到 output_cif 中，这样最后一次就是最终结果
                # 同时返回的 atoms 对象将作为下一步优化的输入
                current_atoms = geo_runner.run(output_path=output_cif, traj_path=step_traj)
            
            # 全部步骤执行成功，更新 yaml 状态
            memory_manager.update_task(name_without_ext, status="completed", geo_params=geo_steps)
            
        except Exception as e:
            print(f"处理 {input_cif} 时发生错误: {e}")
            # 发生异常，记录失败状态，继续处理下一个文件
            memory_manager.update_task(name_without_ext, status="failed", geo_params=geo_steps)

    pass
