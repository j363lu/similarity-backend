FROM nikolaik/python-nodejs:latest
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
RUN pip install pandas
RUN pip install reportlab
RUN pip install openpyxl
EXPOSE 4000
CMD ["npm", "start"]