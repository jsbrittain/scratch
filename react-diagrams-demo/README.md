#react-diagrams-demo

Dockerfile for initialising the [`@projectstorm/react-diagrams`](https://github.com/projectstorm/react-diagrams) demo, which requires more work than suggested on their github page (at least in my experience). The resulting docker image includes depenencies missing from the original resposity, and transfers typescript files from their packages folder to the relevant node_modules folder. There is likely to be a more efficient and elegant solution, but this works for now and allows the demo to run (and thus alteration and extensions to be made).
