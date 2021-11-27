package com.example.mangroveappfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class RehabForAll extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rehab_for_all);
    }
    public void launchSteps(View v) {
        //launch a new activity
        Intent i = new Intent(this, StepsPactices.class);
        startActivity(i);
    }
    public void launcProtect(View v) {
        //launch a new activity
        Intent i = new Intent(this, Protection.class);
        startActivity(i);
    }
    public void launchMonitoring(View v) {
        //launch a new activity
        Intent i = new Intent(this, Monitoring.class);
        startActivity(i);
    }
    public void launchRules(View v) {
        //launch a new activity
        Intent i = new Intent(this, Rules.class);
        startActivity(i);
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
    }
}