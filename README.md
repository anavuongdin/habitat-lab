## Installation for HabiCrowd baselines

Clone the version from our forked github repository and install habitat-lab using the commands below. Note that python>=3.7 is required for working with habitat-lab. All the development and testing was done using python3.7. Please use 3.7 to avoid possible issues.

    ```bash
    git clone https://github.com/habicrowd/habitat-lab.git
    cd habitat-lab
    pip install -e .
    ```

    The command above will install only core of Habitat Lab. To include habitat_baselines along with all additional requirements, use the command below instead:

    ```bash
    git clone https://github.com/habicrowd/habitat-lab.git
    cd habitat-lab
    pip install -r requirements.txt
    python setup.py develop --all # install habitat and habitat_baselines
    ```
