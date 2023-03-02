FROM node:latest
WORKDIR /app
COPY * /app/

# yarn init (interactive)
# index.html
# tsconfig.json
# webpack.config.js

# Dev dependencies (for interactivity)
# RUN apt update
# RUN apt install vim

# Yarn dependencies (put in json file)
# RUN yarn add @projectstorm/react-diagrams
# RUN yarn add webpack
# RUN yarn add -D webpack-cli
# RUN yarn add -D webpack-dev-server
# RUN yarn add html-webpack-plugin
# RUN yarn add ts-loader typescript
# RUN yarn add source-map-loader style-loader babel-loader css-loader
# RUN yarn add babel-core @babel/core @babel/preset-env
# RUN yarn add react react-dom

RUN yarn

# Stop build warnings - copy source files from projectstorm packages/ folder to relevant node_modules folder.
# There must be a better way to resolve this, but works for now.
RUN git clone https://github.com/projectstorm/react-diagrams.git
RUN cp -r react-diagrams/packages/* node_modules/@projectstorm/

RUN yarn
CMD ["yarn", "start" ]
