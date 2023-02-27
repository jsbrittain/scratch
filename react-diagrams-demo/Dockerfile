FROM node:latest
WORKDIR /app
RUN git clone https://github.com/projectstorm/react-diagrams.git
WORKDIR /app/react-diagrams/diagrams-demo-project
RUN yarn add @projectstorm/react-diagrams
RUN yarn add ts-loader typescript @babel/preset-env
# Copies ts files from packages/ folder to relevant node_modules folder.
# This must surely be achievable by other means, but works until I find it.
RUN cp -r ../packages/* node_modules/@projectstorm
RUN yarn
CMD ["yarn", "start" ]
