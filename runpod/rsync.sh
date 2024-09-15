rsync -rvz -e 'ssh -p 24573 -i ~/.ssh/runpod' --progress ../llm root@213.181.110.225:/workspace
rsync -rvz -e 'ssh -p 24573 -i ~/.ssh/runpod' --progress ../runpod root@213.181.110.225:/workspace
rsync -rvz -e 'ssh -p 24573 -i ~/.ssh/runpod' --progress ../pyproject.toml root@213.181.110.225:/workspace