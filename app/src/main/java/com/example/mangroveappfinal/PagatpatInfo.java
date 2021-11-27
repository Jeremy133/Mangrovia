package com.example.mangroveappfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class PagatpatInfo extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pagatpat_info);
    }

    public void launchRehab(View v) {
        //launch a new activity
        Intent i = new Intent(this, RehabForAll.class);
        startActivity(i);

    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();

    }
}