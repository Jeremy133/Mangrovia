package com.example.mangroveappfinal;

public class Model {

    String bakhaw,pagatpat,bungalon,total;

    public String getBakhaw() {
        return bakhaw;
    }

    public String getPagatpat() {
        return pagatpat;
    }

    public String getBungalon() {
        return bungalon;
    }

    public String getTotal() {
        if(bakhaw == null){
            return  "";
        }
        return total =  Integer.parseInt(bungalon) + Integer.parseInt(pagatpat) +  Integer.parseInt(bakhaw) + "" ;   }
}
