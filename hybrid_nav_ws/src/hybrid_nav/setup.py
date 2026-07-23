from setuptools import setup
package_name = "hybrid_nav"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/runsim.launch.py"]),  # <-- ÖNEMLİ
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Hybrid adaptive navigation (beginner friendly).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = hybrid_nav.perception_node:main",
            "adaptation_node = hybrid_nav.adaptation_node:main",
        ],
    },
)
