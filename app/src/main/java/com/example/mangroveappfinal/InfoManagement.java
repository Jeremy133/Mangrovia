package com.example.mangroveappfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class InfoManagement extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info_management);
    }

    public void launchBungaloninfo(View v) {
        //launch a new activity
        Intent i = new Intent(this, BungalonInfo.class);
        startActivity(i);
    }

    public void launchPagatpatinfo(View v) {
        //launch a new activity
        Intent i = new Intent(this, PagatpatInfo.class);
        startActivity(i);

    }

    public void launchBakhawinfo(View v) {
        //launch a new activity
        Intent i = new Intent(this, BakhawInfo.class);
        startActivity(i);

    }

    public void launchRehab(View v) {
        //launch a new activity
        Intent i = new Intent(this, RehabForAll.class);
        startActivity(i);

    }
    public void backPress(View V){
        onBackPressed();
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
    }
}
