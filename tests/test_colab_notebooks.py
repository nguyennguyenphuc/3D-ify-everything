import json
import re
from pathlib import Path
import nbformat
from colab import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_current_and_valid():
    for name, nb in (('agent_bootstrap.ipynb', build_notebook.agent_bootstrap()), ('courtyard_artifixer.ipynb', build_notebook.full_pipeline()),
                     ('video_tour.ipynb', build_notebook.video_tour())):
        nbformat.validate(nb)
        saved = nbformat.read(ROOT/'colab'/name, as_version=4)
        assert [c.source for c in saved.cells] == [c.source for c in nb.cells], f'{name} lệch; chạy python colab/build_notebook.py'


def test_code_cells_compile_without_magics():
    for name in ('agent_bootstrap.ipynb', 'courtyard_artifixer.ipynb', 'video_tour.ipynb'):
        nb = nbformat.read(ROOT/'colab'/name, as_version=4)
        for cell in nb.cells:
            if cell.cell_type != 'code': continue
            code = '\n'.join('pass' if re.match(r'\s*[!%]', line) else line for line in cell.source.splitlines())
            compile(code, name, 'exec')


def test_pipeline_commands_in_notebooks_parse():
    from colab.pipeline import build_parser
    parser = build_parser()
    parser.parse_args(['run', '--source', 'courtyard', '--name', 'courtyard', '--target-frames', '32', '--train-image-max', '1536',
                       '--reconstruction-steps', '10000', '--artifixer3d-steps', '30000', '--trajectory-frames', '81', '--low-memory'])
    parser.parse_args(['run', '--source', 'video', '--input', '/x.mov', '--name', 'n', '--target-frames', '32',
                       '--reconstruction-steps', '10000', '--artifixer3d-steps', '30000', '--trajectory-frames', '81',
                       '--low-memory', '--stages', 'inputs'])
    parser.parse_args(['setup'])
    parser.parse_args(['check'])


def test_video_tour_has_papermill_parameters():
    nb = nbformat.read(ROOT/'colab'/'video_tour.ipynb', as_version=4)
    tagged = [c for c in nb.cells if 'parameters' in c.metadata.get('tags', [])]
    assert len(tagged) == 1 and 'VIDEO =' in tagged[0].source
    stages = [s for s in ('inputs', 'vggt', 'artifixer', 'package') if any(f'"--stages", "{s}"' in c.source for c in nb.cells)]
    assert stages == ['inputs', 'vggt', 'artifixer', 'package']
