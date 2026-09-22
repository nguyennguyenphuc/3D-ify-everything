"""Fail closed: require Metal compiler, compile define, metallib, binary fingerprint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from studio.config import OPENSPLAT


def main():
    import torch
    xcode = "/Applications/Xcode.app/Contents/Developer"
    if os.path.isdir(xcode): os.environ.setdefault("DEVELOPER_DIR", xcode)
    subprocess.run(['xcrun','-sdk','macosx','metal','--version'],check=True)
    source = OPENSPLAT.parent.parent
    code = source/'opensplat.cpp'
    text = code.read_text()
    if 'STUDIO_METAL_REQUIRED' not in text:
        text = text.replace('torch::Device device = torch::kCPU;', '''// STUDIO_METAL_REQUIRED
#ifndef USE_MPS
    std::cerr << "STUDIO_ERROR Metal build required" << std::endl;
    return EXIT_FAILURE;
#endif
    if (!torch::hasMPS() || result.count("cpu")) {
        std::cerr << "STUDIO_ERROR MPS device required" << std::endl;
        return EXIT_FAILURE;
    }
    torch::Device device = torch::kCPU;''')
        text = text.replace('for (; step <= numIters; step++){', '''auto studioStart = std::chrono::steady_clock::now();
        const char* studioBudgetEnv = std::getenv("STUDIO_TRAIN_SECONDS");
        double studioBudget = studioBudgetEnv ? std::stod(studioBudgetEnv) : 1800.;
        for (; step <= numIters; step++){
            if (std::chrono::duration<double>(std::chrono::steady_clock::now()-studioStart).count() >= studioBudget) {
                model.save(outputScene, step - 1);
                std::cout << "STUDIO_BUDGET " << step - 1 << std::endl;
                return EXIT_SUCCESS;
            }
            auto studioBefore = model.means.detach().clone();''')
        text = text.replace('model.schedulersStep(step);', '''model.schedulersStep(step);
            if (step % 10 == 0) {
                auto studioLoss = mainLoss.item<float>();
                if (!std::isfinite(studioLoss)) throw std::runtime_error("Nonfinite loss");
                bool changed = studioBefore.sizes() != model.means.sizes() || !torch::equal(studioBefore, model.means);
                std::cout << "STUDIO_UPDATE " << step << " " << changed << std::endl;
            }''')
        text = '#include <chrono>\n#include <cstdlib>\n#include <cmath>\n' + text
        code.write_text(text)
    cmake = str(root/'.venv/bin/cmake')
    prefix = torch.utils.cmake_prefix_path + ';/opt/homebrew'
    subprocess.run([cmake,'-S',str(source),'-B',str(OPENSPLAT.parent),'-G','Ninja',f'-DCMAKE_MAKE_PROGRAM={root}/.venv/bin/ninja',f'-DCMAKE_PREFIX_PATH={prefix}','-DGPU_RUNTIME=MPS','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],check=True)
    commands = (OPENSPLAT.parent/'compile_commands.json').read_text()
    if '-DUSE_MPS' not in commands: raise RuntimeError('Build không có USE_MPS; từ chối CPU fallback')
    subprocess.run([cmake,'--build',str(OPENSPLAT.parent),'-j','4'],check=True)
    if not list(OPENSPLAT.parent.rglob('*.metallib')): raise RuntimeError('Thiếu metallib')
    receipt = {'backend':'MPS','torch':torch.__version__,'sha256':hashlib.sha256(OPENSPLAT.read_bytes()).hexdigest(),'upstream':json.loads((root/'upstream.lock.json').read_text())['OpenSplat'],'patch_sha256':hashlib.sha256(code.read_bytes()).hexdigest()}
    (OPENSPLAT.parent/'metal-build.json').write_text(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
